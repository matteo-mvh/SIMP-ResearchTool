"""Native equilibrium continuation with lightweight stable-cycle detection.

1D:
    Native equilibrium continuation and bifurcation classification.
    Stable periodic attractors detected through adaptive ODE integration
    and fixed-parameter shooting, without collocation meshes.

2D:
    Existing native equilibrium and periodic continuation workflow.

All ecological equations are imported from model/shared_eqn.py.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from collections import Counter

import csv
import json
import time
import re
import os
import multiprocessing as mp
import traceback
import pickle
import tempfile

import numpy as np

from scipy.integrate import solve_ivp
from scipy.optimize import root

from app.core.control import load_args, StopRequested
from model.continuation_config import parse_job

# ==============================================================
# Stable periodic-attractor search settings
# ==============================================================

# Number of ordinary scan locations follows x_resolution.
# Additional locations are inserted into oscillatory-unstable intervals.

EXTRA_OSCILLATORY_POINTS = 7

# Number of generic interior starting states per parameter location.
GENERIC_SEEDS = 2

# Forward integration is used to obtain an initial periodic guess.
INITIAL_CYCLES = 6
MAX_INITIAL_DAYS = 20000.
DEFAULT_PERIOD = 400.

# Numerical ODE integration tolerances.
RTOL = 1e-7
ATOL = 1e-9

# Independent, tighter integration for final verification.
VERIFY_RTOL = 1e-9
VERIFY_ATOL = 1e-11

# Shooting and Floquet validation.
SHOOTING_TOL = 1e-6
FLOQUET_TOL = 1e-4
MIN_AMPLITUDE = 1e-3

# Reject unrealistic or numerically degenerate periods.
MIN_PERIOD = .1
MAX_PERIOD = 1e5

# ==============================================================
# General output utilities
# ==============================================================

def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text('', encoding='utf-8')
        return

    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({
            key: json.dumps(value)
            if isinstance(value, (list, dict, tuple))
            else value
            for key, value in row.items()
        } for row in rows)


def _atomic_bytes(path, data):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_bytes(data)
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return

        except PermissionError:
            if attempt == 7:
                raise

            time.sleep(.025*(attempt+1))


def write_json(path, payload):
    _atomic_bytes(path, json.dumps(payload, indent=2, allow_nan=False,).encode('utf-8'),)

# ==============================================================
# Model helpers
# ==============================================================

def state_of(row):
    return np.array([float(row[c]) for c in 'MSP'])


def parameters_at(base, axes, uv):
    return replace(base, **{axis.name: axis.physical(value) for axis, value in zip(axes, uv)},)


def physical_trajectory(states, p):
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[0] != 3:
        return False

    if not np.all(np.isfinite(states)):
        return False

    if np.min(states) < 0:
        return False

    nitrogen = (p.N_wc-states[0]-p.H*states[2])
    tolerance = 1e-8*max(1., p.N_wc)
    return bool(np.min(nitrogen) >= -tolerance)


def feasible_seed(state, p):
    if p.N_wc <= 0:
        return None

    state = np.maximum(np.asarray(state, dtype=float), 1e-10,)
    if not np.all(np.isfinite(state)):
        return None

    inventory = state[0]+p.H*state[2]
    if inventory >= .98*p.N_wc:
        factor = .98*p.N_wc/inventory
        state[0] *= factor
        state[2] *= factor

    return state

# ==============================================================
# Direct integration in logarithmic biomass coordinates
# ==============================================================

def log_rhs(t, z, p):
    """Integrate the original equations while preserving positivity."""
    from model.shared_eqn import per_capita
    if not np.all(np.isfinite(z)):
        raise ValueError('Nonfinite logarithmic biomass')

    if np.max(z) > 100.:
        raise ValueError('Unbounded logarithmic biomass')

    y = np.exp(z)
    result = per_capita(y, p)
    if not np.all(np.isfinite(result)):
        raise ValueError('Nonfinite growth rate')

    return result


def log_jacobian(z, p):
    """Jacobian of dz/dt = f(exp(z))/exp(z)."""
    from model.shared_eqn import jacobian, per_capita
    y = np.exp(z)
    if np.min(y) <= 1e-15:
        raise ValueError('Biomass too small for logarithmic Jacobian')

    F = per_capita(y, p)
    J = jacobian(y, p)
    return ((J-np.diag(F)) * y[None, :] / y[:, None])




def equilibrium_distance(state, equilibrium):
    scale = np.maximum(np.abs(equilibrium), 1e-6)
    return float(np.max(np.abs(state-equilibrium)/scale))


def near_stable_equilibrium(state, equilibria):
    return any(
        equilibrium_distance(
            state,
            np.asarray(e.biomass),
        ) < 1e-3
        for e in equilibria
        if e.stability.classification == 'stable'
    )

# ==============================================================
# Estimate period from the equilibrium Jacobian
# ==============================================================

def estimate_period(equilibria):
    candidates = []
    for e in equilibria:
        if e.community != 'MSP':
            continue

        for value in e.stability.eigenvalues:
            if value.imag > 1e-8:
                candidates.append((value.real, 2*np.pi/value.imag,))

    if not candidates:
        return DEFAULT_PERIOD

    # Prefer the most unstable or least damped complex mode.
    _, period = max(candidates, key=lambda item: item[0],)
    return float(np.clip(period, MIN_PERIOD, MAX_PERIOD,))

# ==============================================================
# Obtain a periodic guess from a forward trajectory
# ==============================================================

# Poincare return and local solver settings (not global search limits).
RETURN_TOL = 2e-7
RETURN_REL_TOL = .005
RETURN_EXTRA_CYCLES = 12
POINCARE_MAXFEV = 32


def profiled_ivp(fun, span, y0, diagnostic, stage, heartbeat=None, **kwargs):
    """Time every IVP, including failed evaluations and slow live calls."""
    started = time.monotonic()
    calls, rhs_seconds, last = 0, 0., started
    item = dict(stage=stage, requested_days=float(span[1]-span[0]),
                dimension=len(y0), method=kwargs.get('method', 'LSODA'))
    diagnostic['active_ivp'] = item

    def wrapped(t, y):
        nonlocal calls, rhs_seconds, last
        before = time.monotonic()
        calls += 1
        try:
            return fun(t, y)
        finally:
            now = time.monotonic()
            rhs_seconds += now-before
            if heartbeat and now-last >= 2.:
                item.update(seconds=now-started, rhs_calls=calls, rhs_seconds=rhs_seconds,
                            integrated_days=float(t-span[0]))
                heartbeat(diagnostic)
                last = now

    solution = None
    try:
        solution = solve_ivp(wrapped, span, y0, **kwargs)
        if not solution.success:
            raise RuntimeError(solution.message)
        return solution
    except BaseException as exc:
        item['error'] = str(exc)
        raise
    finally:
        item.update(seconds=time.monotonic()-started, rhs_calls=calls, rhs_seconds=rhs_seconds)
        if solution is not None:
            dt = np.diff(solution.t)
            item.update(nfev=int(solution.nfev), njev=int(solution.njev or 0),
                        nlu=int(solution.nlu or 0), steps=len(dt), success=bool(solution.success),
                        integrated_days=float(solution.t[-1]-span[0]),
                        min_step=float(dt.min()) if len(dt) else None,
                        max_step=float(dt.max()) if len(dt) else None,
                        message=str(solution.message))
        totals = diagnostic.setdefault('ivp_totals', {})
        group = totals.setdefault(stage, {})
        for key in ('seconds', 'rhs_calls', 'rhs_seconds', 'nfev', 'njev', 'nlu', 'steps'):
            group[key] = group.get(key, 0)+item.get(key, 0)
        group['solves'] = group.get('solves', 0)+1
        diagnostic['last_ivp'] = dict(item)
        slow = diagnostic.setdefault('slowest_ivps', [])
        slow.append(dict(item))
        slow.sort(key=lambda record: record['seconds'], reverse=True)
        del slow[5:]
        diagnostic.pop('active_ivp', None)


def trajectory_guess(p, state, equilibria, diagnostic=None, heartbeat=None, period_hint=None):
    """Use successive downward crossings of dlog(P)/dt=0 (P maxima)."""
    diagnostic = {} if diagnostic is None else diagnostic
    state = feasible_seed(state, p)
    if state is None:
        diagnostic['trajectory_reason'] = 'infeasible_seed'
        return None
    T = float(period_hint or estimate_period(equilibria))
    T = float(np.clip(T, MIN_PERIOD, MAX_PERIOD))
    duration = min(MAX_INITIAL_DAYS, max(1000., INITIAL_CYCLES*T))
    z = np.log(state)

    def peak(t, z):
        return log_rhs(t, z, p)[2]
    peak.direction = -1
    peak.terminal = False

    def integrate(z, days):
        sol = profiled_ivp(lambda t, y: log_rhs(t, y, p), (0., days), z,
                           diagnostic, 'trajectory', heartbeat, method='LSODA',
                           rtol=VERIFY_RTOL, atol=VERIFY_ATOL, dense_output=True,
                           events=peak, max_step=max(MIN_PERIOD, T/30))
        if not physical_trajectory(np.exp(sol.y), p):
            raise ValueError('nonphysical_trajectory')
        return sol

    def inspect(sol):
        times, states = sol.t_events[0], sol.y_events[0]
        mask = times > 1e-5
        times, states = times[mask], states[mask]
        diagnostic['peak_count'] = len(times)
        if len(times) < 4:
            diagnostic['trajectory_reason'] = 'insufficient_returns'
            return None
        periods = np.diff(times[-4:])
        Tlast = float(periods[-1])
        returns = np.max(np.abs(np.diff(states[-4:], axis=0)), axis=1)
        sampled = np.exp(sol.sol(np.linspace(times[-2], times[-1], 257)))
        amplitude = float(np.max(np.ptp(sampled, axis=1)/np.maximum(sampled.mean(axis=1), 1e-8)))
        drift = float(np.max(np.abs(periods/Tlast-1))) if Tlast > 0 else 1.
        diagnostic.update(return_errors=returns.tolist(), period_drift=drift,
                          candidate_period=Tlast, candidate_amplitude=amplitude)
        if not MIN_PERIOD <= Tlast <= MAX_PERIOD or drift > .2:
            diagnostic['trajectory_reason'] = 'inconsistent_return_period'
            return None
        if amplitude < MIN_AMPLITUDE or np.min(sampled) <= 1e-8:
            diagnostic['trajectory_reason'] = 'stationary_or_boundary_candidate'
            return None
        if returns[-1] > .5:
            diagnostic['trajectory_reason'] = 'insufficient_recurrence'
            return None
        converged = bool(np.max(returns[-2:]) <= RETURN_TOL and drift <= 1e-5)
        diagnostic['trajectory_reason'] = 'recurrent' if converged else 'approximate_return'
        return dict(log_state=states[-1], period=Tlast, recurrence=float(returns[-1]),
                    converged=converged, amplitude=amplitude)

    sol = integrate(z, duration)
    guess = inspect(sol)
    # Only extend a genuinely contracting return sequence. Very slow convergence
    # near Hopf is handled by local correction rather than thousands of cycles.
    if guess and not guess['converged']:
        errors = diagnostic['return_errors']
        if errors[-2] > 0 and 0 < errors[-1]/errors[-2] < .85:
            ratio = errors[-1]/errors[-2]
            needed = max(1, int(np.ceil(np.log(RETURN_TOL/errors[-1])/np.log(ratio))))
            extra = min(RETURN_EXTRA_CYCLES, needed+3)*guess['period']
            extra = min(extra, MAX_INITIAL_DAYS-duration)
            if needed <= RETURN_EXTRA_CYCLES and extra >= 4*guess['period']:
                improved = inspect(integrate(sol.y[:, -1], extra))
                if improved is not None:
                    guess = improved
    return guess

# ==============================================================
# Fixed-parameter shooting
# ==============================================================

def analytical_cycle_exclusion(p):
    """Sufficient conditions excluding strictly positive MSP cycles.

    A positive periodic biomass requires the mean per-capita
    growth rate over one period to be zero.

    If a species has strictly negative growth everywhere in
    the physical domain, an MSP periodic orbit is impossible.

    An empty result means inconclusive, NOT that a cycle exists.
    """
    if p.N_wc <= 0:
        return 'Zero available nitrogen inventory'

    Nmax = p.N_wc / p.H

    # Maximum available light, attained without biological shading.
    I_bottom = p.I_0 * np.exp(-p.kw * p.H)
    I_water = p.I_0 * np.exp(-p.kw * p.H / 2)

    # Upper bounds on per-capita growth.
    # Grazing and interspecific mortality are nonnegative and
    # can therefore be omitted when constructing upper bounds.
    F_M_max = (p.mu_M * I_bottom / (p.K_I_M + I_bottom) * Nmax / (p.K_N_M + Nmax) - p.m_M)
    F_S_max = (p.mu_S * I_bottom / (p.K_I_S + I_bottom) - p.m_S)
    F_P_max = (p.mu_P * I_water / (p.K_I_P + I_water) * Nmax / (p.K_N_P + Nmax) - p.m_P - p.g_P)
    bounds = {'M': F_M_max, 'S': F_S_max, 'P': F_P_max,}

    # Keep a conservative numerical margin.
    for species, upper in bounds.items():
        if upper < -1e-10:
            return (f'{species}: maximum per-capita growth ' f'{upper:.8g} d^-1 is negative')

    return None


def hopf_initial_guesses(data, base, axis, u, equilibria):
    """Generate initial shooting guesses from verified Hopf points.

    The classifier uses scaled biomass coordinates and
    Kuznetsov's first Lyapunov coefficient:

        l1 = Re(G21) / (2*omega)

    Consequently, the leading radial normal form is

        dr/dt = alpha*r + omega*l1*r**3.

    The leading cycle amplitude is therefore

        r = sqrt(-alpha / (omega*l1)).

    These are initial guesses only. Every accepted orbit must
    still pass shooting correction and independent verification.
    """
    from model.shared_eqn import jacobian
    from scripts.bifurcation_classification import (critical_eigenvectors,)
    candidates = [e for e in equilibria if e.community == 'MSP']
    if not candidates:
        return []

    p = parameters_at(base, [axis], [u])
    guesses = []
    events = sorted(
        [
            event
            for event in data.get('events', [])
            if (
                event.get('kind') == 'hopf'
                and event.get('community') == 'MSP'
                and event.get('classification_status') == 'verified'
                and event.get('classification') == 'hopf_supercritical'
            )
        ],
        key=lambda event: abs(float(event['u'])-u),
    )
    for event in events:
        try:
            uh = float(event['u'])
            du = u - uh

            # The normal form is a local approximation.
            if abs(du) > max(.002, 2/axis.resolution):
                continue

            omega = float(event['omega'])
            l1 = float(event['first_lyapunov_coefficient'])
            crossing = float(event.get('unfolding_u', event['parameter_coefficient'],))
            if (omega <= 0 or l1 >= 0 or crossing*du <= 0):
                continue

            if int(event.get('noncritical_unstable_directions', 0)) != 0:
                continue

            yh = state_of(event)

            # Reproduce the classifier's state normalization.
            scale = np.maximum(np.abs(yh), 1.)
            ph = parameters_at(base, [axis], [uh])
            A = (jacobian(yh, ph) * scale[None, :] / scale[:, None])
            q, _ = critical_eigenvectors(A, 1j*omega)

            # Use the actual neighboring equilibrium rather
            # than assuming its biomass remains fixed.
            equilibrium = min(
                candidates,
                key=lambda e: np.linalg.norm(
                    (np.asarray(e.biomass)-yh)/scale
                ),
            )
            y = np.asarray(equilibrium.biomass, dtype=float,)
            if np.max(abs(y-yh)/scale) > .5:
                continue

            eigenvalues = np.asarray(equilibrium.stability.eigenvalues)
            pair = [value for value in eigenvalues if value.imag > 1e-8]
            if not pair:
                continue

            critical = min(pair, key=lambda value: abs(value.imag-omega),)
            alpha = float(critical.real)
            if alpha <= 0:
                continue

            r_squared = -alpha/(omega*l1)
            if r_squared <= 0:
                continue

            r = np.sqrt(r_squared)
            period = float(2*np.pi/critical.imag)
            if not MIN_PERIOD <= period <= MAX_PERIOD:
                continue

            # Reject guesses outside the local neighborhood.
            relative_size = np.max(2*r*scale*np.abs(q) / np.maximum(y, 1e-8))
            if relative_size > .55:
                continue

            # Different phases represent the same leading
            # periodic orbit. Select feasible initial states.
            for phase in (0., np.pi/2):
                displacement = (2*r*scale * np.real(q*np.exp(1j*phase)))
                state = y + displacement
                if np.min(state) <= 0:
                    continue

                if not physical_trajectory(state[:, None], p):
                    continue

                guesses.append(dict(
                    log_state=np.log(state),
                    period=period,
                    hopf_u=uh,
                    predicted_radius=float(r),
                    predicted_relative_amplitude=float(
                        relative_size
                    ),
                ))

        except (KeyError, TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError,):
            continue

    return guesses




def correct_cycle(p, guess, diagnostic=None, heartbeat=None):
    """Fallback shooting: two-dimensional transverse Poincare return map.

    Use only the three original ODEs during root correction; variational
    equations are reserved for independent Floquet verification.
    """
    diagnostic = {} if diagnostic is None else diagnostic
    reference = np.asarray(guess['log_state'], float)
    T = float(guess['period'])
    tangent = log_rhs(0., reference, p)
    norm = np.linalg.norm(tangent)
    if norm < 1e-9:
        diagnostic['correction_reason'] = 'stationary_phase_section'
        return None
    normal = tangent/norm
    basis = np.linalg.svd(normal[None, :], full_matrices=True)[2][1:].T
    cache = {}
    evaluations = 0

    def section(t, z):
        return float(normal @ (z-reference))
    section.direction = 1
    section.terminal = True

    def evaluate(q):
        nonlocal evaluations
        key = tuple(float(v) for v in q)
        if key in cache:
            return cache[key]
        evaluations += 1
        diagnostic['shooting_evaluations'] = evaluations
        if not np.all(np.isfinite(q)) or np.linalg.norm(q) > 3.:
            raise ValueError('shooting_left_local_section')
        z0 = reference+basis@q
        if not physical_trajectory(np.exp(z0[:, None]), p):
            raise ValueError('nonphysical_shooting_seed')
        # Advance away from t=0 so the section event is a genuine return.
        advance = min(.01*T, .1)
        common = dict(method='LSODA', rtol=VERIFY_RTOL, atol=VERIFY_ATOL,
                      max_step=max(MIN_PERIOD, T/40))
        fun = lambda t, z: log_rhs(t, z, p)
        first = profiled_ivp(fun, (0., advance), z0, diagnostic, 'return_map', heartbeat, **common)
        sol = profiled_ivp(fun, (advance, 1.6*T), first.y[:, -1], diagnostic,
                           'return_map', heartbeat, events=section, **common)
        if not len(sol.t_events[0]):
            raise ValueError('no_transverse_return')
        period = float(sol.t_events[0][0])
        if not .5*T <= period <= 1.6*T or not MIN_PERIOD <= period <= MAX_PERIOD:
            raise ValueError('unexpected_return_period')
        z1 = sol.y_events[0][0]
        if not physical_trajectory(np.exp(sol.y), p):
            raise ValueError('nonphysical_return_map')
        residual = basis.T@(z1-z0)
        closure = float(np.max(np.abs(z1-z0)))
        cache[key] = residual, z0, period, closure
        diagnostic['last_return_residual'] = closure
        return cache[key]

    try:
        result = root(lambda q: evaluate(q)[0], np.zeros(2), method='hybr',
                      options=dict(xtol=1e-8, eps=1e-8, maxfev=POINCARE_MAXFEV))
        residual, z0, period, closure = evaluate(result.x)
        diagnostic.update(root_success=bool(result.success), root_status=int(result.status),
                          root_message=str(result.message), final_return_residual=closure)
        # MINPACK's R factor estimates conditioning of the local return equation.
        if hasattr(result, 'r') and len(result.r) == 3:
            a, b, c = result.r
            condition = float(np.linalg.cond([[a, b], [0., c]]))
            diagnostic['return_condition'] = condition if np.isfinite(condition) else None
        if closure > RETURN_TOL:
            diagnostic['correction_reason'] = 'return_residual_too_large'
            return None
        diagnostic['correction_reason'] = 'corrected'
        return dict(log_state=z0, period=period, shooting_residual=closure,
                    shooting_evaluations=evaluations, method='poincare_shooting')
    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        diagnostic['correction_reason'] = str(exc)
        return None

# ==============================================================
# Floquet stability from the analytical Jacobian
# ==============================================================

def verify_cycle(p, cycle, diagnostic=None, heartbeat=None):
    """Independent closure, nonstationarity, feasibility and Floquet checks."""
    from model.periodic_math import floquet_classification
    diagnostic = {} if diagnostic is None else diagnostic
    z0 = np.asarray(cycle['log_state'], float)
    T = float(cycle['period'])
    initial = np.r_[z0, np.eye(3).ravel()]

    def reject(reason):
        diagnostic['verification_reason'] = reason
        return None

    def variational_rhs(t, values):
        z = values[:3]
        before = time.monotonic()
        J = log_jacobian(z, p)
        diagnostic['variational_jacobian_seconds'] = (
            diagnostic.get('variational_jacobian_seconds', 0.)+time.monotonic()-before)
        return np.r_[log_rhs(t, z, p), (J@values[3:].reshape(3, 3)).ravel()]

    try:
        sol = profiled_ivp(variational_rhs, (0., T), initial, diagnostic, 'floquet', heartbeat,
                           method='LSODA', rtol=VERIFY_RTOL, atol=VERIFY_ATOL,
                           dense_output=True, max_step=max(MIN_PERIOD, T/50))
        if not np.all(np.isfinite(sol.y)):
            return reject('nonfinite_verification')
        closure = float(np.max(abs(sol.y[:3, -1]-z0)))
        diagnostic['verified_closure'] = closure
        if closure > 5*SHOOTING_TOL:
            return reject('independent_closure_failed')
        multipliers = np.linalg.eigvals(sol.y[3:, -1].reshape(3, 3))
        stability = floquet_classification(multipliers, FLOQUET_TOL)
        diagnostic['floquet_check'] = stability
        if stability['stability'] != 'stable':
            return reject('floquet_'+stability['stability'])
        states = np.exp(sol.sol(np.linspace(0., T, 513))[:3])
        if not physical_trajectory(states, p) or np.min(states) <= 1e-8:
            return reject('nonphysical_or_boundary_cycle')
        amplitude = float(np.max(np.ptp(states, axis=1)/np.maximum(states.mean(axis=1), 1e-8)))
        if amplitude < MIN_AMPLITUDE:
            return reject('collapsed_stationary_solution')
        # Near a weakly attracting cycle, a tiny closure error can hide a large
        # distance from the true fixed point. This is a conservative numerical
        # heuristic, not a rigorous error bound for a nonnormal map.
        radius = stability['spectral_radius']
        relative_drift = closure/max(1.-radius, 1e-12)/amplitude
        diagnostic['contraction_scaled_closure'] = float(relative_drift)
        if relative_drift > RETURN_REL_TOL:
            return reject('weak_contraction_requires_correction')
        nitrogen = p.N_wc-states[0]-p.H*states[2]
        area = np.vstack((states[0], states[1], p.H*states[2], nitrogen))
        row = dict(status='verified_periodic', method=cycle.get('method', 'poincare_return'),
                   community='MSP', period=T, relative_amplitude=amplitude, closure_error=closure,
                   shooting_residual=cycle.get('shooting_residual', closure),
                   nitrogen_min=float(nitrogen.min()), **stability)
        for name, values in zip(('M', 'S', 'P_area', 'N_area'), area):
            row.update({name+'_min': float(values.min()), name+'_max': float(values.max()),
                        name+'_mean': float(values[:-1].mean())})
        row['cycle_log_state'] = z0.tolist()  # Preserve phase for the offline 3D viewer.
        diagnostic['verification_reason'] = 'verified_stable_cycle'
        return dict(row=row, seed=dict(log_state=z0.tolist(), period=T))
    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        return reject(str(exc))

# ==============================================================
# Deterministic initial conditions
# ==============================================================

def initial_states(p, equilibria):
    """Search near every discovered MSP equilibrium and
    at several generic interior initial conditions.
    """
    seeds = []
    for e in equilibria:
        if e.community != 'MSP':
            continue

        state = np.asarray(e.biomass)

        # Perturb the equilibrium in opposite directions.
        seeds.append(state*np.array([1.3, .8, .8]))
        seeds.append(state*np.array([.7, 1.2, 1.2]))

    # Use the highest available seagrass equilibrium as a scale.
    seagrass_scale = max([float(e.biomass[1]) for e in equilibria]+[1.])
    generic = [
        np.array([
            .15*p.N_wc,
            .3*seagrass_scale,
            .15*p.N_wc/p.H,
        ]),
        np.array([
            .40*p.N_wc,
            .8*seagrass_scale,
            .10*p.N_wc/p.H,
        ]),
    ]
    seeds.extend(generic[:GENERIC_SEEDS])
    unique = []
    for state in seeds:
        state = feasible_seed(state, p)
        if state is None:
            continue

        if any(np.allclose(state, old, rtol=1e-4, atol=1e-8,) for old in unique):
            continue

        unique.append(state)

    return unique

# ==============================================================
# Search parameter locations
# ==============================================================

def scan_coordinates(data, base, axis, discovery_points=17):
    """Construct a coarse global grid with dense promising intervals.

    The ordinary discovery resolution is used across the full
    parameter domain. The requested continuation resolution
    determines sampling density in oscillatory-unstable regions.

    Subsequent refinement is based on actual attractor results.
    """
    from model.shared_eqn import jacobian
    resolution = int(axis.resolution)
    discovery_points = max(3, int(discovery_points))
    coordinates = list(np.linspace(0., 1., discovery_points,))
    intervals = []
    centers = []
    priority = []
    for path in data.get('paths', []):
        if not path:
            continue

        if path[0]['community'] != 'MSP':
            continue

        active = []
        previous_u = None
        for row in path:
            u = float(row['u'])
            try:
                p = parameters_at(base, [axis], [u])
                eigenvalues = np.linalg.eigvals(jacobian(state_of(row), p))
                unstable_complex = any(
                    value.real > 1e-8
                    and abs(value.imag) > 1e-8
                    for value in eigenvalues
                )

            except (ValueError, ArithmeticError, np.linalg.LinAlgError,):
                unstable_complex = False

            if unstable_complex:
                if (not active and previous_u is not None):
                    active.append(previous_u)

                active.append(u)

            elif active:
                active.append(u)
                intervals.append((min(active), max(active),))
                active = []

            previous_u = u

        if active:
            intervals.append((min(active), max(active),))

    # Densely sample promising intervals, regardless of their
    # width relative to the total continuation domain.
    for a, b in intervals:
        a = max(0., a)
        b = min(1., b)
        if b < a:
            continue

        count = max(EXTRA_OSCILLATORY_POINTS, int(np.ceil((b-a)*resolution))+1,)
        coordinates.extend(np.linspace(a, b, count))
        centers.append(.5*(a+b))

    # Include locations on both sides of classified Hopf points.
    for event in data.get('events', []):
        if event.get('kind') != 'hopf':
            continue

        if event.get('community') != 'MSP':
            continue

        uh = float(event['u'])
        offset = min(.0004, .1/resolution,)
        coordinates.extend((uh, uh-offset, uh+offset, uh-.5/resolution, uh+.5/resolution,))
        if (
            event.get('classification_status') == 'verified'
            and event.get('classification') == 'hopf_supercritical'
        ):
            crossing = float(event.get('parameter_coefficient', 0.))
            if crossing != 0:

                # Search first on the side where a stable
                # small-amplitude cycle is predicted to exist.
                direction = np.sign(crossing)
                priority.extend((uh+direction*offset, uh+direction*.5/resolution,))

    def valid_unique(values):
        return list(dict.fromkeys(round(float(u), 10) for u in values if 0 <= u <= 1))

    coordinates = valid_unique(coordinates)
    priority = valid_unique(priority)
    targets = priority or centers
    if targets:
        coordinates.sort(key=lambda u: min(abs(u-center) for center in targets))

    else:
        coordinates.sort()

    # Hopf-derived guesses are attempted first.
    return valid_unique(priority+coordinates)


def adaptive_refinement_coordinates(results, axis):
    """Refine detected cycle regions and classification transitions.

    Negative searches alone are not absence certificates.

    Refinement terminates at the selected continuation resolution,
    rather than at an arbitrary global number of attempts.
    """
    ordered = sorted(results, key=lambda row: row['u'],)
    new = []
    nominal_step = 1/max(1, axis.resolution)

    def signature(row):
        return (
            bool(row['stable_cycle_found']),
            row.get('stable_equilibria', ''),
            int(row.get('stable_equilibrium_count', 0)),
            bool(row.get('oscillatory_instability')),
            bool(row.get('analytical_exclusion')),
            bool(row.get('equilibrium_search_error')),
        )

    for a, b in zip(ordered, ordered[1:]):
        ua = float(a['u'])
        ub = float(b['u'])
        gap = ub-ua
        if gap <= nominal_step+1e-9:
            continue

        either_cycle = (a['stable_cycle_found'] or b['stable_cycle_found'])
        either_oscillatory = (a['oscillatory_instability'] or b['oscillatory_instability'])
        transition = (signature(a) != signature(b))

        # No need to refine a cycle search interval if both
        # sampled endpoints have analytical exclusions and
        # no other classification difference.
        both_excluded = (a.get('analytical_exclusion') and b.get('analytical_exclusion'))
        if both_excluded and not transition:
            continue

        if (either_cycle or either_oscillatory or transition):
            new.append(round(.5*(ua+ub), 10))

    return list(dict.fromkeys(new))

# ==============================================================
# Search for a stable attractor at one parameter location
# ==============================================================

def search_location(p, u, equilibria, warm_seed, checkpoint, record_attempt,
                    hopf_guesses=None, live=None, at_hopf=False):
    """Try recurrent trajectories first, then correct only unresolved candidates."""
    tasks, fallback = [], []
    if warm_seed is not None:
        tasks.append(('reuse', np.exp(warm_seed['log_state']), warm_seed['period'], warm_seed.get('source_u')))
    if not at_hopf:
        tasks.extend(('hopf_normal_form', np.exp(g['log_state']), g['period'], None)
                     for g in hopf_guesses or [])
    tasks.extend(('manual', state, None, None) for state in initial_states(p, equilibria))

    def attempt(kind, state, period, parent, guess=None):
        diagnostic = dict(kind=kind, uv=[float(u)])
        started = time.monotonic()
        diagnostic.update(guess_seconds=0., correction_seconds=0., verification_seconds=0.)
        found = None

        def heartbeat(info):
            if live:
                live(info)
            checkpoint(f'{kind} · {info.get("active_ivp", {}).get("stage", "search")}')

        try:
            heartbeat(diagnostic)
            if guess is None:
                before = time.monotonic()
                guess = trajectory_guess(p, state, equilibria, diagnostic, heartbeat, period)
                diagnostic['guess_seconds'] = time.monotonic()-before
                if guess and guess['converged']:
                    before = time.monotonic()
                    found = verify_cycle(p, dict(guess, method='poincare_return'), diagnostic, heartbeat)
                    diagnostic['verification_seconds'] = time.monotonic()-before
                if guess is not None and found is None:
                    # Peak-based guesses at the same location have a common phase.
                    # Do not correct nearly identical return candidates repeatedly.
                    duplicate = any(abs(np.log(guess['period']/g['period'])) < .02 and
                                    np.max(abs(guess['log_state']-g['log_state'])) < .005
                                    for _, _, g in fallback)
                    if not duplicate:
                        fallback.append((kind, parent, guess))
                    else:
                        diagnostic['trajectory_reason'] = 'duplicate_return_candidate'
            else:
                before = time.monotonic()
                corrected = correct_cycle(p, guess, diagnostic, heartbeat)
                diagnostic['correction_seconds'] = time.monotonic()-before
                if corrected is not None:
                    before = time.monotonic()
                    found = verify_cycle(p, corrected, diagnostic, heartbeat)
                    diagnostic['verification_seconds'] = time.monotonic()-before
            diagnostic['reason'] = ('verified_stable_cycle' if found else
                                    diagnostic.get('verification_reason') or diagnostic.get('correction_reason') or
                                    diagnostic.get('trajectory_reason', 'not_confirmed'))
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
            diagnostic['reason'] = str(exc)
        finally:
            diagnostic['seconds'] = time.monotonic()-started
            diagnostic.setdefault('shooting_evaluations', 0)
            record_attempt(diagnostic)
        if found:
            found['row'].update(cycle_parent_u=parent, cycle_search_method=kind,
                                shooting_evaluations=diagnostic['shooting_evaluations'])
        return found

    for kind, state, period, parent in tasks:
        found = attempt(kind, state, period, parent)
        if found:
            return found
    # All independent trajectories have been tested before expensive correction.
    # Most recurrent candidates are attempted first; no global attempt cap.
    for kind, parent, guess in sorted(fallback, key=lambda entry: entry[2]['recurrence']):
        found = attempt('poincare_shooting_'+kind, None, None, parent, guess)
        if found:
            return found
    return None

# ==============================================================
# Lightweight 1D periodic-attractor scan
# ==============================================================

def analyze_stable_attractors(data, base, axis, checkpoint, folder, options=None,):
    from model.shared_eqn import solve_equilibria
    options = options or {}
    folder = Path(folder)
    started = time.monotonic()
    coordinates = scan_coordinates(
        data,
        base,
        axis,
        discovery_points=int(
            options.get('discovery_points', 17)
        ),
    )
    coarse_count = len(coordinates)
    results = []
    attempts = []
    bank = []
    scheduled = set(coordinates)
    refinement_round = 0
    payload = dict(
        # Keep the existing method name for plotting compatibility.
        method='Adaptive ODE integration and shooting',
        algorithm='SIMP Poincare return-first search',
        status='running',
        settings=dict(
            scan_points=len(coordinates),
            coarse_points=coarse_count,
            scan_resolution=int(axis.resolution),
            analytical_screening=True,
            analytical_shooting_jacobian=False,
            return_first=True,
            poincare_shooting=True,
            hopf_normal_form_initialization=True,
            adaptive_sampling=True,
            generic_seeds=GENERIC_SEEDS,
            initial_cycles=INITIAL_CYCLES,
            max_initial_days=MAX_INITIAL_DAYS,
            rtol=RTOL,
            atol=ATOL,
            verification_rtol=VERIFY_RTOL,
            verification_atol=VERIFY_ATOL,
            floquet_tolerance=FLOQUET_TOL,
        ),
        locations=results,
        sampling_coordinates=[],
        attempts=attempts,
        branches=[],
        hopf=[],
        coverage={},
    )
    data['periodic_analysis'] = payload
    data['stable_orbits'] = []
    data['oscillations'] = []
    suffix = f'H{base.H:g}'

    def save():
        completed = len(results)
        confirmed = sum(row['stable_cycle_found'] for row in results)
        excluded = sum(bool(row['analytical_exclusion']) for row in results)
        unresolved = [
            [row['u']]
            for row in results
            if (
                not row['stable_cycle_found']
                and not row['analytical_exclusion']
            )
        ]
        payload['sampling_coordinates'] = [float(u) for u in coordinates]
        payload['settings']['scan_points'] = (len(coordinates))
        payload['coverage'].update(
            scheduled_locations=len(coordinates),
            checked_locations=completed,
            completed_locations=completed,
            native_attempts=0,
            search_attempts=len(attempts)-excluded,
            shooting_attempts=sum(a.get('kind', '').startswith('poincare_shooting_') for a in attempts),
            direct_return_cycles=sum(r.get('method') == 'poincare_return' for r in results),
            analytical_exclusions=excluded,
            refinement_rounds=refinement_round,
            elapsed_seconds=time.monotonic()-started,
            verified_periodic_locations=confirmed,
            unvisited_locations=[
                [u]
                for u in coordinates[completed:]
            ],
            incomplete_locations=unresolved,
            pending_branches=[],
            incomplete_arms=[],
            limit_reached=completed<len(coordinates),
        )
        payload['timing'] = dict(
            guess_seconds=sum(
                a.get('guess_seconds', 0.)
                for a in attempts
            ),
            correction_seconds=sum(
                a.get('correction_seconds', 0.)
                for a in attempts
            ),
            verification_seconds=sum(
                a.get('verification_seconds', 0.)
                for a in attempts
            ),
        )
        ordered = sorted(results, key=lambda row: row['u'],)
        data['stable_orbits'] = [
            row
            for row in ordered
            if (
                row['stable_cycle_found']
                and row['status'] == 'verified_periodic'
                and row['stability'] == 'stable'
            )
        ]

        # Preserve the existing periodic-envelope interface.
        data['oscillations'] = list(data['stable_orbits'])
        write_json(folder/f'periodic_{suffix}.json', payload,)
        write_csv(folder/f'stable_orbit_scan_{suffix}.csv', ordered,)

    def record_attempt(attempt):
        attempts.append(attempt)
        payload.pop('active_attempt', None)
        save()

    def live_attempt(attempt):
        payload['active_attempt'] = attempt
        save()

    save()
    try:
        i = 0
        while True:

            # --------------------------------------------------
            # Evaluate all currently scheduled locations.
            # --------------------------------------------------
            while i < len(coordinates):
                u = coordinates[i]
                stage = ('coarse' if i < coarse_count else 'refinement')
                prefix = (f'Periodic stable scan ' f'{i+1}/{len(coordinates)}' f' · {stage}')
                checkpoint(prefix)
                location_started = time.monotonic()
                p = parameters_at(base, [axis], [u])
                x = float(axis.physical(u))

                # ----------------------------------------------
                # Analytical screening
                # ----------------------------------------------
                exclusion = (analytical_cycle_exclusion(p) or '')

                # Equilibrium analysis remains necessary even
                # where positive periodic cycles are excluded.
                equilibrium_error = ''
                try:
                    equilibria = solve_equilibria(p)

                except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError,) as exc:
                    equilibria = []
                    equilibrium_error = str(exc)

                stable_equilibria = [
                    e
                    for e in equilibria
                    if (
                        e.stability.classification
                        == 'stable'
                    )
                ]
                stable_communities = sorted({e.community for e in stable_equilibria})
                oscillatory_instability = any(
                    e.community == 'MSP'
                    and any(
                        value.real > 1e-8
                        and abs(value.imag) > 1e-8
                        for value in e.stability.eigenvalues
                    )
                    for e in equilibria
                )
                found = None
                if exclusion:

                    # A sufficient mathematical exclusion:
                    # no forward integration or shooting needed.
                    attempts.append(dict(
                        kind='analytical_exclusion',
                        uv=[float(u)],
                        seconds=0.,
                        reason=exclusion,
                    ))

                else:

                    # ------------------------------------------
                    # Reuse neighboring verified cycles.
                    # ------------------------------------------
                    warm_seed = None
                    if bank:
                        nearest = min(bank, key=lambda entry: abs(entry['u']-u),)
                        if abs(nearest['u']-u) <= max(.02, 3/axis.resolution,):
                            warm_seed = dict(nearest['seed'], source_u=float(nearest['u']),)

                    # ------------------------------------------
                    # Construct local Hopf approximations.
                    # ------------------------------------------
                    hopf_guesses = hopf_initial_guesses(data, base, axis, u, equilibria,)

                    # ------------------------------------------
                    # Search using cheapest methods first.
                    # ------------------------------------------
                    found = search_location(
                        p,
                        u,
                        equilibria,
                        warm_seed,
                        checkpoint=lambda message: checkpoint(
                            prefix+' · '+message
                        ),
                        record_attempt=record_attempt,
                        hopf_guesses=hopf_guesses,
                        live=live_attempt,
                        at_hopf=any(e.get('kind') == 'hopf' and e.get('community') == 'MSP'
                                    and e.get('classification_status') == 'verified'
                                    and abs(float(e['u'])-u) < 1e-8
                                    for e in data.get('events', [])),
                    )

                stable_cycle = found is not None
                stable_eq = bool(stable_equilibria)
                if stable_cycle and stable_eq:
                    classification = 'coexistence'

                elif stable_cycle and equilibrium_error:
                    classification = ('stable_cycle_equilibria_unresolved')

                elif stable_cycle:
                    classification = ('stable_cycle_no_stable_equilibrium_found')

                elif stable_eq:
                    classification = ('stable_equilibrium_cycle_not_confirmed')

                else:
                    classification = 'unresolved'

                row = dict(
                    H=float(base.H),
                    u=float(u),
                    uv=[float(u)],
                    x=x,
                    stable_equilibria='+'.join(
                        stable_communities
                    ),
                    stable_equilibrium_count=len(
                        stable_equilibria
                    ),
                    equilibrium_search_error=(
                        equilibrium_error
                    ),
                    oscillatory_instability=(
                        oscillatory_instability
                    ),
                    analytical_exclusion=exclusion,
                    stable_cycle_found=stable_cycle,
                    coexistence_eq_and_cycle=(
                        stable_cycle and stable_eq
                    ),
                    attractor_class=classification,
                    sampling_stage=stage,
                    status=(
                        'verified_periodic'
                        if stable_cycle
                        else 'analytically_excluded'
                        if exclusion
                        else 'cycle_not_confirmed'
                    ),
                    seconds=(
                        time.monotonic()-location_started
                    ),
                )
                if found is not None:
                    row.update(found['row'])
                    bank.append(dict(u=float(u), seed=found['seed'],))

                results.append(row)
                i += 1
                save()
                checkpoint(prefix+' complete')

            # --------------------------------------------------
            # Adaptively refine promising parameter intervals.
            # --------------------------------------------------
            additional = [
                u
                for u in adaptive_refinement_coordinates(
                    results, axis
                )
                if u not in scheduled
            ]
            if not additional:
                break

            refinement_round += 1
            if bank:
                additional.sort(key=lambda u: min(abs(u-entry['u']) for entry in bank))

            coordinates.extend(additional)
            scheduled.update(additional)
            save()
            checkpoint(
                f'Adaptive periodic refinement '
                f'{refinement_round} · '
                f'{len(additional)} additional locations'
            )

        payload['status'] = (
            'completed_with_unresolved_searches'
            if any(
                not row['stable_cycle_found']
                and not row['analytical_exclusion']
                for row in results
            )
            else 'completed'
        )

    except BaseException:
        payload['status'] = 'interrupted'
        raise

    finally:
        save()

    summary = dict(
        method=payload['method'],
        algorithm=payload['algorithm'],
        locations=len(results),
        analytical_exclusions=sum(
            bool(row['analytical_exclusion'])
            for row in results
        ),
        stable_periodic_locations=sum(
            row['stable_cycle_found']
            for row in results
        ),
        coexistence_locations=sum(
            row['coexistence_eq_and_cycle']
            for row in results
        ),
        classifications=dict(Counter(
            row['attractor_class']
            for row in results
        )),
        attempt_types=dict(Counter(
            a['kind']
            for a in attempts
        )),
        refinement_rounds=refinement_round,
        attempts=len(attempts),
        timing=payload['timing'],
        seconds=time.monotonic()-started,
        limitations=(
            'Analytical exclusions apply to strictly positive '
            'MSP periodic orbits. Other unsuccessful searches '
            'are inconclusive. Coarse sampling can miss narrow '
            'disconnected periodic-attractor regions. '
            'Unstable cycles and exact periodic folds are not traced.'
        ),
    )
    write_json(folder/f'stable_orbit_summary_{suffix}.json', summary,)
    plot_stable_attractors(data, axis, folder/'figures',)
    return results

# ==============================================================
# Stable-attractor visualization
# ==============================================================

def plot_stable_attractors(data, axis, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scripts.continuation_plots import axis_label
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    results = sorted(
        data.get(
            'periodic_analysis', {}
        ).get('locations', []),
        key=lambda row: row['x'],
    )
    if not results:
        return

    colors = {
        'coexistence': '#8055b5',
        'stable_cycle_no_stable_equilibrium_found': '#df8531',
        'stable_cycle_equilibria_unresolved': '#b69c68',
        'stable_equilibrium_cycle_not_confirmed': '#278360',
        'unresolved': '#999999',
    }
    labels = {
        'coexistence': 'Stable equilibrium + stable cycle',
        'stable_cycle_no_stable_equilibrium_found':
            'Stable cycle; no stable equilibrium found',
        'stable_cycle_equilibria_unresolved':
            'Stable cycle; equilibria unresolved',
        'stable_equilibrium_cycle_not_confirmed':
            'Stable equilibrium; cycle not confirmed',
        'unresolved': 'Unresolved',
    }
    fig, panels = plt.subplots(
        3,
        1,
        figsize=(11, 9),
        sharex=True,
        layout='constrained',
        gridspec_kw=dict(
            height_ratios=[3., 2., 1.]
        ),
    )
    biomass, stability, states = panels

    # ----------------------------------------------------------
    # Seagrass equilibrium branches and verified orbit amplitudes
    # ----------------------------------------------------------
    for path in data.get('paths', []):
        for a, b in zip(path[:-1], path[1:]):
            stable = (a['stability'] == b['stability'] == 'stable')
            biomass.plot(
                [a['x'], b['x']],
                [a['S'], b['S']],
                color='#278360',
                lw=1.5,
                ls='-' if stable else ':',
            )

    verified = [row for row in results if row['stable_cycle_found']]
    for row in verified:
        color = colors[row['attractor_class']]
        biomass.vlines(row['x'], row['S_min'], row['S_max'], color=color, lw=2.2,)
        biomass.scatter(row['x'], row['S_mean'], color=color, s=12, zorder=5,)
        stability.scatter(row['x'], row['spectral_radius'], color=color, s=22,)

    biomass.set_ylabel('Seagrass [mmol N m$^{-2}$]')
    biomass.set_ylim(bottom=0)
    biomass.legend(
        handles=[
            Line2D(
                [], [],
                color='#278360',
                label='Stable equilibrium',
            ),
            Line2D(
                [], [],
                color='#278360',
                ls=':',
                label='Unstable / neutral equilibrium',
            ),
            Line2D(
                [], [],
                color='#df8531',
                marker='|',
                ls='',
                label='Verified periodic biomass range',
            ),
        ],
        frameon=False,
        fontsize=9,
    )

    # ----------------------------------------------------------
    # Stability of verified cycles
    # ----------------------------------------------------------
    stability.axhline(1., color='#444444', ls='--', lw=1.,)
    stability.set_ylabel('Nontrivial Floquet\nspectral radius')
    stability.set_ylim(bottom=0, top=1.05,)

    # ----------------------------------------------------------
    # Attractor classification at sampled locations
    # ----------------------------------------------------------
    for row in results:
        states.scatter(row['x'], 0., marker='s', s=26, color=colors[row['attractor_class']],)

    states.set_yticks([])
    states.set_ylim(-.4, .4)
    states.set_ylabel('Attractors')
    states.set_xlabel(axis_label(axis))
    used = {row['attractor_class'] for row in results}
    states.legend(
        handles=[
            Line2D(
                [], [],
                marker='s',
                ls='',
                color=color,
                label=labels[key],
            )
            for key, color in colors.items()
            if key in used
        ],
        loc='upper center',
        bbox_to_anchor=(.5, -.65),
        ncol=2,
        fontsize=8,
        frameon=False,
    )
    for ax in panels:
        ax.set_xlim(axis.low, axis.high)
        ax.grid(alpha=.15)

    fig.suptitle(f'SIMP stable attractors · H = {data["H"]:g} m')

    # Do not interpolate between separately tested cycle locations.
    name = (folder / f'stable_attractors_{axis.name}_H{data["H"]:g}')
    fig.savefig(Path(str(name)+'.png'), dpi=270, bbox_inches='tight',)
    fig.savefig(Path(str(name)+'.pdf'), bbox_inches='tight',)
    plt.close(fig)

# ==============================================================
# Equilibrium continuation for one water depth
# ==============================================================

def _run_depth(p, axes, options, dimensions, folder, report):
    from model.pyoomph_continuation import (
        continue_1d,
        continue_2d,
        locate_events,
        native_session,
    )
    from model.continuation_refinement import (refine_regions,)
    depth = p.H
    started = time.monotonic()
    local_progress = 0.
    current_start = ''

    def checkpoint(message):
        nonlocal local_progress, current_start
        counts = re.search(r'(\d+)/(\d+)', message)
        if counts and 'Equilibrium overview' in message:
            local_progress = max(local_progress, .15*int(counts[1])/int(counts[2]),)

        elif counts and 'Checking start' in message:
            local_progress = max(local_progress, .15+.60*int(counts[1])/int(counts[2]),)
            current_start = counts[0]

        elif counts and 'Checking region' in message:
            local_progress = max(local_progress, .75+.20*int(counts[1])/int(counts[2]),)

        elif 'correction' in message:
            local_progress = max(local_progress, .76 if dimensions == 1 else .15,)

        elif 'Native 2D' in message:
            local_progress = max(local_progress, .45)

        phase = f'H={depth:g} m · {message}'
        if (current_start and message.startswith('Native ') and dimensions == 1):
            phase += f' · start {current_start}'

        report(.75*local_progress, phase,)

    # ----------------------------------------------------------
    # Native equilibrium continuation
    # ----------------------------------------------------------
    with native_session(
        reuse=(
            dimensions == 2
            and str(
                options.get('reuse_native', 'off')
            ) == 'on'
        ),
        step_factor=(
            float(
                options.get('adaptive_step_factor', 1)
            )
            if dimensions == 2 else 1
        ),
    ) as stats:
        common = dict(
            discovery_points=int(
                options['discovery_points']
            ),
            max_steps=int(
                options['max_steps']
            ),
            checkpoint=checkpoint,
        )
        if dimensions == 1:
            data = continue_1d(p, axes[0], **common)
            if (
                any(
                    r['community'] != '0'
                    for r in data['scan']
                )
                and not any(
                    path
                    and path[0]['community'] != '0'
                    and len(path) > 1
                    for path in data['paths']
                )
            ):
                raise RuntimeError('No native branches could be traced. ' 'Inspect the worker log.')

            events = []
            for path in data['paths']:
                found, notes = locate_events(
                    p,
                    axes,
                    [[row] for row in path],
                    0,
                    checkpoint,
                    branch_trace=True,
                )
                for event in found:
                    if not any(
                        e['kind'] == event['kind']
                        and e['community'] == event['community']
                        and e['invader'] == event['invader']
                        and abs(e['u']-event['u']) < 1e-5
                        for e in events
                    ):
                        events.append(event)

                data['diagnostics'].extend(notes)

            data['events'] = events
            rows = [
                dict(
                    row,
                    branch=branch,
                    point=i,
                    method=(
                        'exact_empty'
                        if row['community'] == '0'
                        else 'pyoomph_arclength'
                    ),
                )
                for branch, path in enumerate(
                    data['paths'], 1
                )
                for i, row in enumerate(path)
            ]
            write_csv(folder/f'branches_H{depth:g}.csv', rows,)

        else:
            discovery_start = time.monotonic()
            data = continue_2d(
                p,
                axes,
                transects=int(
                    options['transects']
                ),
                discovery_refinement=int(
                    options['discovery_refinement']
                ),
                **common,
            )
            discovery_seconds = (time.monotonic()-discovery_start)
            refinement_start = time.monotonic()

            def save_refinement_checkpoint(current, history):
                write_csv(
                    folder/f'curves_H{depth:g}.csv',
                    [
                        dict(
                            row,
                            curve=curve,
                            point=i,
                            method='pyoomph_event_arclength',
                        )
                        for curve, path in enumerate(
                            current['curves'], 1
                        )
                        for i, row in enumerate(path)
                    ],
                )
                write_json(
                    folder/f'refinement_checkpoint_H{depth:g}.json',
                    dict(
                        rounds=history,
                        curve_count=len(
                            current['curves']
                        ),
                        status='partial_checkpoint',
                    ),
                )

            save_refinement_checkpoint(data, [])
            data = refine_regions(
                data,
                p,
                axes,
                int(options['region_samples']),
                int(options['max_steps']),
                checkpoint,
                max_rounds=int(
                    options['refinement_rounds']
                ),
                searches_per_round=int(
                    options['refinement_searches']
                ),
                on_round=save_refinement_checkpoint,
            )
            refinement_seconds = (time.monotonic()-refinement_start)
            write_json(folder/f'geometry_H{depth:g}.json', data.get('geometry_diagnostics', []),)
            write_json(folder/f'refinement_H{depth:g}.json', data['refinement'],)
            rows = [
                dict(
                    row,
                    curve=curve,
                    point=i,
                    method='pyoomph_event_arclength',
                )
                for curve, path in enumerate(
                    data['curves'], 1
                )
                for i, row in enumerate(path)
            ]
            write_csv(folder/f'curves_H{depth:g}.csv', rows,)
            write_csv(
                folder/f'region_samples_H{depth:g}.csv',
                [
                    dict(
                        s['selected'] or {},
                        region=i,
                        label=r['label'],
                        u=s['u'],
                        v=s['v'],
                        signature=s['signature'],
                    )
                    for i, r in enumerate(
                        data['regions'], 1
                    )
                    for s in r['samples']
                ],
            )
            write_json(
                folder/f'regions_H{depth:g}.json',
                [
                    dict(
                        label=r['label'],
                        normalized_polygon_wkt=(
                            r['polygon'].wkt
                        ),
                        area=r['polygon'].area,
                        signatures=r['signatures'],
                        reasons=r['reasons'],
                    )
                    for r in data['regions']
                ],
            )

        write_csv(folder/f'overview_H{depth:g}.csv', data['scan'],)
        write_csv(folder/f'events_H{depth:g}.csv', data['events'],)
        write_json(folder/f'diagnostics_H{depth:g}.json', data['diagnostics'],)

    # ----------------------------------------------------------
    # Classify bifurcations before the periodic search
    # ----------------------------------------------------------
    report(.74, f'H={depth:g} m · Preparing equilibrium previews',)
    try:
        from scripts.continuation_preflight import (prepare_equilibrium_results,)
        prepare_equilibrium_results(data, p, axes, folder,)

    except StopRequested:
        raise

    except Exception:
        traceback.print_exc()
        (
            folder
            / f'equilibrium_preview_error_H{depth:g}.txt'
        ).write_text(
            traceback.format_exc(),
            encoding='utf-8',
        )

    stats['equilibrium_seconds'] = (time.monotonic()-started)

    # ----------------------------------------------------------
    # Periodic analysis
    # ----------------------------------------------------------
    oscillation_start = time.monotonic()
    periodic_progress = .75

    def periodic_checkpoint(message):
        nonlocal periodic_progress
        if message.startswith('Periodic stable scan'):
            counts = re.search(r'(\d+)/(\d+)', message)
            if counts:
                current = int(counts[1])
                total = max(1, int(counts[2]))
                completed = (current if message.endswith('complete') else current-1)
                fraction = completed/total
                estimate = (.75+.10*fraction if '· coarse' in message else .85+.13*fraction)
                periodic_progress = max(periodic_progress, min(estimate, .98),)

        elif message.startswith('Periodic Hopf seeds'):
            counts = re.search(r'(\d+)/(\d+)', message)
            if counts:
                fraction = ((int(counts[1])-1) / max(int(counts[2]), 1))
                periodic_progress = max(periodic_progress, .75+.06*fraction,)

        elif message.startswith('Periodic discovery'):
            counts = re.search(r'(\d+)/(\d+).*round (\d+)/(\d+)', message,)
            if counts:
                current, total, round_index, rounds = map(int, counts.groups())
                fraction = (round_index-1 + (current-1)/max(total, 1))/max(rounds, 1)
                periodic_progress = max(periodic_progress, .82+.17*fraction,)

        report(min(periodic_progress, .99), f'H={depth:g} m · {message}',)

    periodic_checkpoint('Starting automatic periodic analysis')
    try:
        if dimensions == 1:

            # Lightweight stable-attractor detection.
            # No PyOOMPH periodic mesh or periodic branch tracking.
            analyze_stable_attractors(
                data,
                p,
                axes[0],
                periodic_checkpoint,
                folder,
                options=options,
            )

        else:

            # Keep the established 2D implementation unchanged.
            from model.periodic_orbits import (analyze_periodic,)
            analyze_periodic(data, p, axes, options, periodic_checkpoint, folder,)

    except StopRequested:
        if dimensions == 1:
            try:
                plot_stable_attractors(data, axes[0], folder/'figures',)

            except Exception:
                traceback.print_exc()

        raise

    except Exception as exc:
        traceback.print_exc()
        error_text = traceback.format_exc()
        payload = data.setdefault('periodic_analysis', {})
        for key in ('branches', 'locations', 'attempts', 'hopf',):
            payload.setdefault(key, [])

        payload.setdefault('coverage', {})
        payload['status'] = 'failed'
        payload['reason'] = str(exc)
        payload['coverage']['limit_reached'] = True
        data.setdefault('oscillations', [])
        data.setdefault('stable_orbits', [])
        for branch in payload['branches']:
            for arm in branch['arms']:
                if arm['termination'] == 'running':
                    arm['termination'] = 'failed'

        try:
            (folder / f'periodic_error_H{depth:g}.txt').write_text(error_text, encoding='utf-8',)
            if dimensions == 1:
                write_json(folder/f'periodic_H{depth:g}.json', payload,)
                plot_stable_attractors(data, axes[0], folder/'figures',)

            else:
                from model.periodic_orbits import (export_periodic,)
                export_periodic(data, folder, depth)

        except Exception:
            traceback.print_exc()

        periodic_checkpoint('Periodic analysis failed · ' 'retaining equilibrium results')

    # ----------------------------------------------------------
    # Save measured performance
    # ----------------------------------------------------------
    stats['oscillation_seconds'] = (time.monotonic()-oscillation_start)
    stats['elapsed_seconds'] = (time.monotonic()-started)
    if dimensions == 2:
        stats.update(discovery_seconds=discovery_seconds, refinement_seconds=refinement_seconds,)

    data['performance'] = stats
    write_json(folder/f'performance_H{depth:g}.json', stats,)
    report(1., f'H={depth:g} m · complete',)
    return data

# ==============================================================
# Parallel depth calculations
# ==============================================================

def _child(p, axes, options, dimensions, folder, work, index,):
    last = 0.

    def report(progress, phase):
        nonlocal last
        now = time.monotonic()
        if now-last >= .5 or progress == 1.:
            _atomic_bytes(work/f'{index}.progress', json.dumps([progress, phase]).encode('utf-8'),)
            last = now

    try:
        data = _run_depth(p, axes, options, dimensions, folder, report,)
        _atomic_bytes(
            work/f'{index}.result',
            pickle.dumps(
                data,
                protocol=pickle.HIGHEST_PROTOCOL,
            ),
        )

    except BaseException:
        _atomic_bytes(work/f'{index}.error', traceback.format_exc().encode('utf-8'),)
        raise


def _parallel_depths(base, depths, axes, options, dimensions, ctx, workers,):
    mpctx = mp.get_context('spawn')
    active = {}
    progress = [0.]*len(depths)
    results = {}
    next_index = 0
    with tempfile.TemporaryDirectory(prefix='simp_workers_', dir=ctx.folder,) as temp:
        work = Path(temp)
        try:
            while len(results) < len(depths):
                while (next_index < len(depths) and len(active) < workers):
                    i = next_index
                    next_index += 1
                    process = mpctx.Process(
                        target=_child,
                        args=(
                            replace(
                                base, H=depths[i]
                            ),
                            axes,
                            options,
                            dimensions,
                            ctx.folder,
                            work,
                            i,
                        ),
                    )
                    keys = (
                        'OMP_NUM_THREADS',
                        'OPENBLAS_NUM_THREADS',
                        'MKL_NUM_THREADS',
                        'BLIS_NUM_THREADS',
                        'NUMEXPR_NUM_THREADS',
                        'PYTHONFAULTHANDLER',
                    )
                    old = {k: os.environ.get(k) for k in keys}
                    try:
                        os.environ.update({k: '1' for k in keys})
                        process.start()

                    finally:
                        for k, v in old.items():
                            if v is None:
                                os.environ.pop(k, None)

                            else:
                                os.environ[k] = v

                    active[i] = process

                phases = []
                for i, process in list(active.items()):
                    path = work/f'{i}.progress'
                    if path.exists():
                        try:
                            fraction, phase = json.loads(path.read_text(encoding='utf-8'))
                            progress[i] = max(progress[i], fraction,)
                            phases.append(phase)

                        except (OSError, ValueError):
                            pass

                    if not process.is_alive():
                        process.join()
                        error = work/f'{i}.error'
                        result = work/f'{i}.result'
                        if (process.exitcode != 0 or not result.exists()):
                            detail = (
                                error.read_text(
                                    encoding='utf-8'
                                )
                                if error.exists()
                                else f'exit code {process.exitcode}'
                            )
                            raise RuntimeError(f'Depth H={depths[i]:g} ' f'worker failed: {detail}')

                        results[i] = pickle.loads(result.read_bytes())
                        progress[i] = 1.
                        process.close()
                        del active[i]

                ctx.checkpoint(
                    5+80*sum(progress)/len(depths),
                    f'Parallel depths '
                    f'{len(results)}/{len(depths)} · '
                    + ' | '.join(phases[:workers]),
                )
                if len(results) < len(depths):
                    time.sleep(.25)

        finally:
            for process in active.values():
                if process.is_alive():
                    process.terminate()

                process.join(timeout=3)
                if process.is_alive():
                    process.kill()
                    process.join()

                process.close()

    return [results[i] for i in range(len(depths))]

# ==============================================================
# Main continuation workflow
# ==============================================================

def run(values, ctx, dimensions):
    ctx.plot_start = 90.
    ctx.checkpoint(0, 'Preparing continuation', force=True,)
    base, depths, axes, options = parse_job(values, dimensions)
    started = time.monotonic()
    workers = (min(len(depths), int(options.get('depth_workers', 1)),) if dimensions == 2 else 1)
    if workers > 1:
        datasets = _parallel_depths(base, depths, axes, options, dimensions, ctx, workers,)

    else:
        datasets = []
        for index, depth in enumerate(depths):

            def report(fraction, phase):
                ctx.checkpoint(5+80*(index+fraction)/len(depths), phase,)

            datasets.append(_run_depth(
                replace(base, H=depth),
                axes,
                options,
                dimensions,
                ctx.folder,
                report,
            ))

    from scripts.continuation_plots import (plot_1d, plot_2d,)
    ctx.checkpoint(90, 'Rendering PNG and PDF figures', force=True,)
    if dimensions == 1:
        plot_1d(datasets, axes[0], ctx.folder/'figures',)

    else:
        plot_2d(datasets, axes, ctx.folder/'figures',)

    summary = dict(
        engine='pyoomph native pseudo-arclength',
        periodic_engine=(
            'adaptive_ode_shooting'
            if dimensions == 1
            else 'native_periodic_bvp'
        ),
        dimensions=dimensions,
        depths=depths,
        axes=[vars(a) for a in axes],
        seconds=time.monotonic()-started,
        equation_source='model/shared_eqn.py',
        limitations=(
            '1D periodic results are confirmed attracting '
            'cycles at independently sampled parameter locations. '
            'Finite initial conditions and parameter sampling '
            'cannot establish global absence of cycles. '
            'Unstable periodic branches and exact periodic folds '
            'are not traced in 1D. '
            '2D native periodic results remain individual '
            'verified locations and transects.'
        ),
        diagnostics=sum(
            len(d['diagnostics'])
            for d in datasets
        ),
        depth_workers=workers,
        performance={
            str(d['H']): d['performance']
            for d in datasets
        },
    )
    summary['periodic'] = {
        str(d['H']): dict(
            verified_points=len(
                d.get('stable_orbits', [])
                if dimensions == 1
                else d.get('oscillations', [])
            ),
            coexistence_points=sum(
                r.get(
                    'coexistence_eq_and_cycle', False
                )
                for r in d.get('stable_orbits', [])
            ),
            status=d.get(
                'periodic_analysis', {}
            ).get('status', 'unavailable'),
            coverage=d.get(
                'periodic_analysis', {}
            ).get('coverage', {}),
        )
        for d in datasets
    }
    if dimensions == 2:
        summary['refinement'] = {str(d['H']): d['refinement'] for d in datasets}
        summary['unresolved_area'] = {
            str(d['H']): sum(
                r['polygon'].area
                for r in d['regions']
                if r['label'] == '?'
            )
            for d in datasets
        }

    write_json(ctx.folder/'summary.json', summary,)
    ctx.checkpoint(100, 'Completed native continuation', force=True,)
    return datasets


def main(dimensions):
    mp.freeze_support()
    values, ctx, job = load_args()
    try:
        run(values, ctx, dimensions)

    except StopRequested:
        return 130

    return 0
