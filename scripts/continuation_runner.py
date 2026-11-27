
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

    fields = list(dict.fromkeys(
        key for row in rows for key in row
    ))

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

    _atomic_bytes(
        path,
        json.dumps(
            payload,
            indent=2,
            allow_nan=False,
        ).encode('utf-8'),
    )


# ==============================================================
# Model helpers
# ==============================================================

def state_of(row):

    return np.array([
        float(row[c]) for c in 'MSP'
    ])


def parameters_at(base, axes, uv):

    return replace(
        base,
        **{
            axis.name: axis.physical(value)
            for axis, value in zip(axes, uv)
        },
    )


def physical_trajectory(states, p):

    states = np.asarray(states, dtype=float)

    if states.ndim != 2 or states.shape[0] != 3:
        return False

    if not np.all(np.isfinite(states)):
        return False

    if np.min(states) < 0:
        return False

    nitrogen = (
        p.N_wc-states[0]-p.H*states[2]
    )

    tolerance = 1e-8*max(1., p.N_wc)

    return bool(np.min(nitrogen) >= -tolerance)


def feasible_seed(state, p):

    if p.N_wc <= 0:
        return None

    state = np.maximum(
        np.asarray(state, dtype=float),
        1e-10,
    )

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
        raise ValueError(
            'Biomass too small for logarithmic Jacobian'
        )

    F = per_capita(y, p)
    J = jacobian(y, p)

    return (
        (J-np.diag(F))
        * y[None, :]
        / y[:, None]
    )


def integrate_log(
    z0,
    duration,
    p,
    rtol=RTOL,
    atol=ATOL,
    dense_output=False,
    events=None,
):

    solution = solve_ivp(
        lambda t, z: log_rhs(t, z, p),
        (0., duration),
        np.asarray(z0, dtype=float),
        method='LSODA',
        rtol=rtol,
        atol=atol,
        dense_output=dense_output,
        events=events,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    if not physical_trajectory(
        np.exp(solution.y), p
    ):
        raise ValueError(
            'Integration left physical domain'
        )

    return solution


def equilibrium_distance(state, equilibrium):

    scale = np.maximum(
        np.abs(equilibrium), 1e-6
    )

    return float(np.max(
        np.abs(state-equilibrium)/scale
    ))


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

                candidates.append((
                    value.real,
                    2*np.pi/value.imag,
                ))

    if not candidates:
        return DEFAULT_PERIOD

    # Prefer the most unstable or least damped complex mode.
    _, period = max(
        candidates,
        key=lambda item: item[0],
    )

    return float(np.clip(
        period,
        MIN_PERIOD,
        MAX_PERIOD,
    ))


# ==============================================================
# Obtain a periodic guess from a forward trajectory
# ==============================================================

def trajectory_guess(p, state, equilibria):

    state = feasible_seed(state, p)

    if state is None:
        return None

    period_guess = estimate_period(equilibria)

    duration = min(
        MAX_INITIAL_DAYS,
        max(
            1000.,
            INITIAL_CYCLES*period_guess,
        ),
    )

    # A maximum of phytoplankton biomass provides a convenient
    # Poincare section: d(log(P))/dt changes from positive to negative.
    def peak(t, z):

        return log_rhs(t, z, p)[2]

    peak.direction = -1
    peak.terminal = False

    solution = integrate_log(
        np.log(state),
        duration,
        p,
        events=peak,
        dense_output=True,
    )

    terminal = np.exp(solution.y[:, -1])

    if near_stable_equilibrium(
        terminal, equilibria
    ):
        return None

    times = solution.t_events[0]
    states = solution.y_events[0]

    # Ignore the initial point if it lies on the section.
    mask = times > 1e-5

    times = times[mask]
    states = states[mask]

    if len(times) < 3:
        return None

    periods = np.diff(times[-3:])

    period = float(periods[-1])

    if not MIN_PERIOD <= period <= MAX_PERIOD:
        return None

    # Candidate intervals should already have similar periods,
    # but full convergence is not necessary before shooting.
    if abs(periods[-1]-periods[-2])/period > .2:
        return None

    z0 = states[-2]

    sampled = np.exp(
        solution.sol(
            np.linspace(
                times[-2],
                times[-1],
                129,
            )
        )
    )

    amplitude = np.max(
        np.ptp(sampled, axis=1)
        / np.maximum(
            np.mean(sampled, axis=1),
            1e-8,
        )
    )

    if amplitude < MIN_AMPLITUDE:
        return None

    # Require recurrence in the complete state, not only P peaks.
    recurrence = float(np.max(
        np.abs(states[-1]-states[-2])
    ))

    if recurrence > .5:
        return None

    return dict(
        log_state=z0,
        period=period,
        initial_recurrence=recurrence,
    )


# ==============================================================
# Fixed-parameter shooting
# ==============================================================

def correct_cycle(p, guess):
    """Solve Phi_T(z0)-z0=0 together with a phase condition.

    Unknowns:
        Three initial logarithmic biomasses and the log-period.

    This is a four-dimensional root problem, irrespective of the
    temporal integration resolution.
    """

    reference = np.asarray(
        guess['log_state'], dtype=float
    )

    period = float(guess['period'])

    tangent = log_rhs(0., reference, p)

    norm = np.linalg.norm(tangent)

    if norm < 1e-9:
        return None

    tangent /= norm

    def residual(values):

        z0 = np.asarray(values[:3])
        T = float(np.exp(values[3]))

        if not MIN_PERIOD <= T <= MAX_PERIOD:
            raise ValueError('Invalid shooting period')

        if np.max(np.abs(z0-reference)) > 5.:
            raise ValueError(
                'Shooting left initial-state neighborhood'
            )

        solution = integrate_log(z0, T, p)

        recurrence = (
            solution.y[:, -1]-z0
        )

        phase = float(
            np.dot(z0-reference, tangent)
        )

        return np.r_[recurrence, phase]

    initial = np.r_[
        reference,
        np.log(period),
    ]

    try:

        result = root(
            residual,
            initial,
            method='hybr',
            options=dict(
                xtol=1e-8,
                maxfev=60,
            ),
        )

        error = float(np.max(
            np.abs(residual(result.x))
        ))

        if error > SHOOTING_TOL:
            return None

        return dict(
            log_state=result.x[:3],
            period=float(np.exp(result.x[3])),
            shooting_residual=error,
        )

    except (
        ValueError,
        RuntimeError,
        ArithmeticError,
        np.linalg.LinAlgError,
    ):

        return None


# ==============================================================
# Floquet stability from the analytical Jacobian
# ==============================================================

def verify_cycle(p, cycle):
    """Independently verify the orbit using tighter integration.

    Integrate:
        dz/dt = F(z)
        dPsi/dt = J_log(z) Psi

    The monodromy matrix has dimension 3 x 3.
    No temporal collocation mesh is required.
    """

    from model.periodic_math import floquet_classification

    z0 = np.asarray(
        cycle['log_state'], dtype=float
    )

    period = float(cycle['period'])

    initial = np.r_[
        z0,
        np.eye(3).ravel(),
    ]

    def variational_rhs(t, values):

        z = values[:3]

        matrix = values[3:].reshape(3, 3)

        return np.r_[
            log_rhs(t, z, p),
            (
                log_jacobian(z, p)@matrix
            ).ravel(),
        ]

    try:

        solution = solve_ivp(
            variational_rhs,
            (0., period),
            initial,
            method='LSODA',
            rtol=VERIFY_RTOL,
            atol=VERIFY_ATOL,
            dense_output=True,
        )

        if not solution.success:
            return None

        if not np.all(np.isfinite(solution.y)):
            return None

        closure_error = float(np.max(
            np.abs(
                solution.y[:3, -1]-z0
            )
        ))

        if closure_error > 5*SHOOTING_TOL:
            return None

        monodromy = solution.y[
            3:, -1
        ].reshape(3, 3)

        multipliers = np.linalg.eigvals(
            monodromy
        )

        stability = floquet_classification(
            multipliers,
            FLOQUET_TOL,
        )

        # Only confirmed attracting periodic orbits are exported.
        if stability['stability'] != 'stable':
            return None

        # Sampling is used only for reporting extrema and means.
        # These are NOT collocation nodes or solver unknowns.
        sample_times = np.linspace(
            0., period, 513
        )

        states = np.exp(
            solution.sol(sample_times)[:3]
        )

        if not physical_trajectory(
            states, p
        ):
            return None

        if np.min(states) <= 1e-8:
            return None

        amplitude = float(np.max(
            np.ptp(states, axis=1)
            / np.maximum(
                np.mean(states, axis=1),
                1e-8,
            )
        ))

        if amplitude < MIN_AMPLITUDE:
            return None

        nitrogen = (
            p.N_wc-states[0]-p.H*states[2]
        )

        area = np.vstack((
            states[0],
            states[1],
            p.H*states[2],
            nitrogen,
        ))

        row = dict(
            status='verified_periodic',
            method='adaptive_shooting',
            community='MSP',
            period=period,
            relative_amplitude=amplitude,
            closure_error=closure_error,
            shooting_residual=cycle[
                'shooting_residual'
            ],
            nitrogen_min=float(
                np.min(nitrogen)
            ),
            **stability,
        )

        for name, values in zip(
            ('M', 'S', 'P_area', 'N_area'),
            area,
        ):

            row[name+'_min'] = float(
                np.min(values)
            )

            row[name+'_max'] = float(
                np.max(values)
            )

            # Equally spaced phase samples approximate cycle means.
            row[name+'_mean'] = float(
                np.mean(values[:-1])
            )

        return dict(
            row=row,
            seed=dict(
                log_state=z0.tolist(),
                period=period,
            ),
        )

    except (
        ValueError,
        RuntimeError,
        ArithmeticError,
        np.linalg.LinAlgError,
    ):

        return None


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
        seeds.append(
            state*np.array([1.3, .8, .8])
        )

        seeds.append(
            state*np.array([.7, 1.2, 1.2])
        )

    # Use the highest available seagrass equilibrium as a scale.
    seagrass_scale = max(
        [
            float(e.biomass[1])
            for e in equilibria
        ]+[1.]
    )

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

    seeds.extend(
        generic[:GENERIC_SEEDS]
    )

    unique = []

    for state in seeds:

        state = feasible_seed(state, p)

        if state is None:
            continue

        if any(
            np.allclose(
                state, old,
                rtol=1e-4,
                atol=1e-8,
            )
            for old in unique
        ):
            continue

        unique.append(state)

    return unique


# ==============================================================
# Search parameter locations
# ==============================================================

def scan_coordinates(data, base, axis):
    """Select a regular grid and enrich oscillatory-unstable intervals."""

    from model.shared_eqn import jacobian

    coordinates = list(
        np.linspace(
            0., 1., axis.resolution+1
        )
    )

    unstable_intervals = []

    for path in data.get('paths', []):

        if not path:
            continue

        if path[0]['community'] != 'MSP':
            continue

        current = []

        for row in path:

            try:

                u = float(row['u'])

                p = parameters_at(
                    base, [axis], [u]
                )

                eigenvalues = np.linalg.eigvals(
                    jacobian(state_of(row), p)
                )

                oscillatory_unstable = any(
                    value.real > 1e-8
                    and abs(value.imag) > 1e-8
                    for value in eigenvalues
                )

            except (
                ValueError,
                ArithmeticError,
                np.linalg.LinAlgError,
            ):

                oscillatory_unstable = False

            if oscillatory_unstable:

                current.append(u)

            elif current:

                unstable_intervals.append(
                    (min(current), max(current))
                )

                current = []

        if current:

            unstable_intervals.append(
                (min(current), max(current))
            )

    centers = []

    for a, b in unstable_intervals:

        coordinates.extend(
            np.linspace(
                a,
                b,
                EXTRA_OSCILLATORY_POINTS,
            )
        )

        centers.append(.5*(a+b))

    # Include verified Hopf locations and points just beside them.
    for event in data.get('events', []):

        if event.get('kind') != 'hopf':
            continue

        if event.get('community') != 'MSP':
            continue

        u = float(event['u'])

        coordinates.extend((
            u,
            u-.1/axis.resolution,
            u+.1/axis.resolution,
        ))

    unique = sorted({
        round(float(u), 10)
        for u in coordinates
        if 0 <= u <= 1
    })

    # Start within promising intervals, then extend outward.
    # This provides warm starts before examining stable-equilibrium
    # regions where disconnected stable cycles might coexist.
    if centers:

        unique.sort(
            key=lambda u: min(
                abs(u-center)
                for center in centers
            )
        )

    return unique


# ==============================================================
# Search for a stable attractor at one parameter location
# ==============================================================

def search_location(
    p,
    u,
    equilibria,
    warm_seed,
    checkpoint,
    record_attempt,
):

    tasks = []

    if warm_seed is not None:

        tasks.append((
            'reuse',
            warm_seed,
        ))

    for state in initial_states(
        p, equilibria
    ):

        tasks.append((
            'manual',
            state,
        ))

    for kind, seed in tasks:

        checkpoint(
            f'Periodic {kind} · u={u:.5f}'
        )

        started = time.monotonic()

        reason = 'not_confirmed'
        found = None

        try:

            if kind == 'reuse':

                # A neighboring verified orbit is usually a much
                # better initial guess than a random perturbation.
                state = feasible_seed(
                    np.exp(seed['log_state']),
                    p,
                )

                if state is not None:

                    guess = dict(
                        log_state=np.log(state),
                        period=seed['period'],
                    )

                else:

                    guess = None

            else:

                guess = trajectory_guess(
                    p,
                    seed,
                    equilibria,
                )

            if guess is not None:

                corrected = correct_cycle(
                    p, guess
                )

                if corrected is not None:

                    found = verify_cycle(
                        p, corrected
                    )

            direct_reuse = (
                kind == 'reuse'
                and found is not None
            )

            # If direct warm-start shooting fails, use forward
            # integration from the previous periodic solution.
            if (
                found is None
                and kind == 'reuse'
                and state is not None
            ):

                guess = trajectory_guess(
                    p, state, equilibria
                )

                if guess is not None:

                    corrected = correct_cycle(
                        p, guess
                    )

                    if corrected is not None:

                        found = verify_cycle(
                            p, corrected
                        )

            if found is not None:
                reason = 'verified_stable_cycle'

        except (
            ValueError,
            RuntimeError,
            ArithmeticError,
            np.linalg.LinAlgError,
        ) as exc:

            reason = str(exc)

        record_attempt(dict(
            kind=kind,
            uv=[float(u)],
            seconds=time.monotonic()-started,
            reason=reason,
        ))

        if found is not None:

            found['row']['cycle_parent_u'] = (
                float(seed['source_u'])
                if (
                    direct_reuse
                    and 'source_u' in seed
                )
                else None
            )

            found['row']['cycle_search_method'] = (
                'direct_warm_start'
                if direct_reuse
                else 'trajectory_initialized_shooting'
            )

            return found

    return None


# ==============================================================
# Lightweight 1D periodic-attractor scan
# ==============================================================

def analyze_stable_attractors(
    data,
    base,
    axis,
    checkpoint,
    folder,
):

    from model.shared_eqn import solve_equilibria

    folder = Path(folder)

    started = time.monotonic()

    coordinates = scan_coordinates(
        data, base, axis
    )

    results = []
    attempts = []
    bank = []

    payload = dict(
        method='Adaptive ODE integration and shooting',
        status='running',
        settings=dict(
            scan_points=len(coordinates),
            scan_resolution=int(axis.resolution),
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
        sampling_coordinates=[
            float(u) for u in coordinates
        ],
        attempts=attempts,

        # No periodic branches are claimed: these are independent
        # verified parameter locations, not full continuation curves.
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

        confirmed = sum(
            row['stable_cycle_found']
            for row in results
        )

        payload['coverage'].update(
            scheduled_locations=len(coordinates),
            checked_locations=completed,
            completed_locations=completed,
            native_attempts=0,
            shooting_attempts=len(attempts),
            elapsed_seconds=time.monotonic()-started,
            verified_periodic_locations=confirmed,
            unvisited_locations=[
                [u] for u in coordinates[completed:]
            ],
            incomplete_locations=[],
            pending_branches=[],
            incomplete_arms=[],
            limit_reached=completed<len(coordinates),
        )

        # Existing diagnostic exporter and ETA reader can access
        # this file using their existing periodic_H*.json pattern.
        write_json(
            folder/f'periodic_{suffix}.json',
            payload,
        )

        ordered = sorted(
            results,
            key=lambda row: row['u'],
        )

        write_csv(
            folder/f'stable_orbit_scan_{suffix}.csv',
            ordered,
        )

        data['stable_orbits'] = [
            row for row in ordered
            if (
                row.get('stable_cycle_found')
                and row.get('status') == 'verified_periodic'
                and row.get('stability') == 'stable'
            )
        ]

        # Existing continuation plots and external consumers
        # use this common periodic-results interface.
        data['oscillations'] = list(
            data['stable_orbits']
        )

    save()

    try:

        for i, u in enumerate(
            coordinates, 1
        ):

            checkpoint(
                f'Periodic stable scan '
                f'{i}/{len(coordinates)}'
            )

            p = parameters_at(
                base, [axis], [u]
            )

            x = float(axis.physical(u))

            equilibrium_error = ''

            try:

                equilibria = solve_equilibria(p)

            except (
                ValueError,
                RuntimeError,
                ArithmeticError,
                np.linalg.LinAlgError,
            ) as exc:

                equilibria = []
                equilibrium_error = str(exc)

            stable_communities = sorted({
                e.community
                for e in equilibria
                if e.stability.classification
                == 'stable'
            })

            oscillatory_instability = any(
                e.community == 'MSP'
                and any(
                    value.real > 1e-8
                    and abs(value.imag) > 1e-8
                    for value in e.stability.eigenvalues
                )
                for e in equilibria
            )

            # Reuse the nearest previously verified cycle.
            warm_seed = None

            if bank:

                nearest = min(
                    bank,
                    key=lambda entry: abs(
                        entry['u']-u
                    ),
                )

                if abs(nearest['u']-u) <= max(
                    .02,
                    3/axis.resolution,
                ):

                    warm_seed = dict(
                        nearest['seed'],
                        source_u=float(nearest['u']),
                    )

            before = time.monotonic()

            found = search_location(
                p,
                u,
                equilibria,
                warm_seed,
                checkpoint=lambda message: checkpoint(
                    f'Periodic stable scan '
                    f'{i}/{len(coordinates)} · '
                    + message
                ),
                record_attempt=attempts.append,
            )

            stable_cycle = found is not None
            stable_eq = bool(stable_communities)

            if stable_cycle and stable_eq:

                classification = 'coexistence'

            elif stable_cycle and equilibrium_error:

                classification = (
                    'stable_cycle_equilibria_unresolved'
                )

            elif stable_cycle:

                classification = (
                    'stable_cycle_no_stable_equilibrium_found'
                )

            elif stable_eq:

                classification = (
                    'stable_equilibrium_cycle_not_confirmed'
                )

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
                stable_equilibrium_count=sum(
                    e.stability.classification
                    == 'stable'
                    for e in equilibria
                ),
                equilibrium_search_error=(
                    equilibrium_error
                ),
                oscillatory_instability=(
                    oscillatory_instability
                ),
                stable_cycle_found=stable_cycle,
                coexistence_eq_and_cycle=(
                    stable_cycle and stable_eq
                ),
                attractor_class=classification,
                status=(
                    'verified_periodic'
                    if stable_cycle
                    else 'cycle_not_confirmed'
                ),
                seconds=time.monotonic()-before,
            )

            if found is not None:

                row.update(found['row'])

                bank.append(dict(
                    u=float(u),
                    seed=found['seed'],
                ))

            results.append(row)

            # Preserve scientific results if a calculation is stopped.
            save()

            checkpoint(
                f'Periodic stable scan '
                f'{i}/{len(coordinates)} complete'
            )

        payload['status'] = (
            'completed_with_unresolved_searches'
            if any(
                not row['stable_cycle_found']
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
        locations=len(results),
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
        attempts=len(attempts),
        seconds=time.monotonic()-started,
        limitations=(
            'Finite initial conditions and finite parameter sampling '
            'do not establish global absence of stable periodic orbits. '
            'Unstable cycles and exact periodic folds are not traced.'
        ),
    )

    write_json(
        folder/f'stable_orbit_summary_{suffix}.json',
        summary,
    )

    plot_stable_attractors(
        data,
        axis,
        folder/'figures',
    )

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

            stable = (
                a['stability'] == b['stability']
                == 'stable'
            )

            biomass.plot(
                [a['x'], b['x']],
                [a['S'], b['S']],
                color='#278360',
                lw=1.5,
                ls='-' if stable else ':',
            )

    verified = [
        row for row in results
        if row['stable_cycle_found']
    ]

    for row in verified:

        color = colors[
            row['attractor_class']
        ]

        biomass.vlines(
            row['x'],
            row['S_min'],
            row['S_max'],
            color=color,
            lw=2.2,
        )

        biomass.scatter(
            row['x'],
            row['S_mean'],
            color=color,
            s=12,
            zorder=5,
        )

        stability.scatter(
            row['x'],
            row['spectral_radius'],
            color=color,
            s=22,
        )

    biomass.set_ylabel(
        'Seagrass [mmol N m$^{-2}$]'
    )

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

    stability.axhline(
        1.,
        color='#444444',
        ls='--',
        lw=1.,
    )

    stability.set_ylabel(
        'Nontrivial Floquet\nspectral radius'
    )

    stability.set_ylim(
        bottom=0,
        top=1.05,
    )

    # ----------------------------------------------------------
    # Attractor classification at sampled locations
    # ----------------------------------------------------------

    for row in results:

        states.scatter(
            row['x'],
            0.,
            marker='s',
            s=26,
            color=colors[
                row['attractor_class']
            ],
        )

    states.set_yticks([])
    states.set_ylim(-.4, .4)

    states.set_ylabel(
        'Attractors'
    )

    states.set_xlabel(
        axis_label(axis)
    )

    used = {
        row['attractor_class']
        for row in results
    }

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

    fig.suptitle(
        f'SIMP stable attractors · H = {data["H"]:g} m'
    )

    # Do not interpolate between separately tested cycle locations.
    name = (
        folder
        / f'stable_attractors_{axis.name}_H{data["H"]:g}'
    )

    fig.savefig(
        Path(str(name)+'.png'),
        dpi=270,
        bbox_inches='tight',
    )

    fig.savefig(
        Path(str(name)+'.pdf'),
        bbox_inches='tight',
    )

    plt.close(fig)


# ==============================================================
# Equilibrium continuation for one water depth
# ==============================================================

def _run_depth(
    p, axes, options, dimensions, folder, report
):

    from model.pyoomph_continuation import (
        continue_1d,
        continue_2d,
        locate_events,
        native_session,
    )

    from model.continuation_refinement import (
        refine_regions,
    )

    depth = p.H
    started = time.monotonic()

    local_progress = 0.
    current_start = ''

    def checkpoint(message):

        nonlocal local_progress, current_start

        counts = re.search(
            r'(\d+)/(\d+)', message
        )

        if counts and 'Equilibrium overview' in message:

            local_progress = max(
                local_progress,
                .15*int(counts[1])/int(counts[2]),
            )

        elif counts and 'Checking start' in message:

            local_progress = max(
                local_progress,
                .15+.60*int(counts[1])/int(counts[2]),
            )

            current_start = counts[0]

        elif counts and 'Checking region' in message:

            local_progress = max(
                local_progress,
                .75+.20*int(counts[1])/int(counts[2]),
            )

        elif 'correction' in message:

            local_progress = max(
                local_progress,
                .76 if dimensions == 1 else .15,
            )

        elif 'Native 2D' in message:

            local_progress = max(
                local_progress, .45
            )

        phase = f'H={depth:g} m · {message}'

        if (
            current_start
            and message.startswith('Native ')
            and dimensions == 1
        ):

            phase += f' · start {current_start}'

        report(
            .75*local_progress,
            phase,
        )

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

            data = continue_1d(
                p, axes[0], **common
            )

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

                raise RuntimeError(
                    'No native branches could be traced. '
                    'Inspect the worker log.'
                )

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

            write_csv(
                folder/f'branches_H{depth:g}.csv',
                rows,
            )

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

            discovery_seconds = (
                time.monotonic()-discovery_start
            )

            refinement_start = time.monotonic()

            def save_refinement_checkpoint(
                current, history
            ):

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

            save_refinement_checkpoint(
                data, []
            )

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

            refinement_seconds = (
                time.monotonic()-refinement_start
            )

            write_json(
                folder/f'geometry_H{depth:g}.json',
                data.get(
                    'geometry_diagnostics', []
                ),
            )

            write_json(
                folder/f'refinement_H{depth:g}.json',
                data['refinement'],
            )

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

            write_csv(
                folder/f'curves_H{depth:g}.csv',
                rows,
            )

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

        write_csv(
            folder/f'overview_H{depth:g}.csv',
            data['scan'],
        )

        write_csv(
            folder/f'events_H{depth:g}.csv',
            data['events'],
        )

        write_json(
            folder/f'diagnostics_H{depth:g}.json',
            data['diagnostics'],
        )

    # ----------------------------------------------------------
    # Classify bifurcations before the periodic search
    # ----------------------------------------------------------

    report(
        .74,
        f'H={depth:g} m · Preparing equilibrium previews',
    )

    try:

        from scripts.continuation_preflight import (
            prepare_equilibrium_results,
        )

        prepare_equilibrium_results(
            data,
            p,
            axes,
            folder,
        )

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

    stats['equilibrium_seconds'] = (
        time.monotonic()-started
    )

    # ----------------------------------------------------------
    # Periodic analysis
    # ----------------------------------------------------------

    oscillation_start = time.monotonic()

    periodic_progress = .75

    def periodic_checkpoint(message):

        nonlocal periodic_progress

        if message.startswith(
            'Periodic stable scan'
        ):

            counts = re.search(
                r'(\d+)/(\d+)', message
            )

            if counts:

                current = int(counts[1])
                total = max(1, int(counts[2]))

                completed = (
                    current
                    if message.endswith('complete')
                    else current-1
                )

                periodic_progress = max(
                    periodic_progress,
                    .75+.24*completed/total,
                )

        elif message.startswith(
            'Periodic Hopf seeds'
        ):

            counts = re.search(
                r'(\d+)/(\d+)', message
            )

            if counts:

                fraction = (
                    (int(counts[1])-1)
                    / max(int(counts[2]), 1)
                )

                periodic_progress = max(
                    periodic_progress,
                    .75+.06*fraction,
                )

        elif message.startswith(
            'Periodic discovery'
        ):

            counts = re.search(
                r'(\d+)/(\d+).*round (\d+)/(\d+)',
                message,
            )

            if counts:

                current, total, round_index, rounds = map(
                    int, counts.groups()
                )

                fraction = (
                    round_index-1
                    + (current-1)/max(total, 1)
                )/max(rounds, 1)

                periodic_progress = max(
                    periodic_progress,
                    .82+.17*fraction,
                )

        report(
            min(periodic_progress, .99),
            f'H={depth:g} m · {message}',
        )

    periodic_checkpoint(
        'Starting automatic periodic analysis'
    )

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
            )

        else:

            # Keep the established 2D implementation unchanged.
            from model.periodic_orbits import (
                analyze_periodic,
            )

            analyze_periodic(
                data,
                p,
                axes,
                options,
                periodic_checkpoint,
                folder,
            )

    except StopRequested:

        if dimensions == 1:

            try:

                plot_stable_attractors(
                    data,
                    axes[0],
                    folder/'figures',
                )

            except Exception:

                traceback.print_exc()

        raise

    except Exception as exc:

        traceback.print_exc()

        error_text = traceback.format_exc()

        payload = data.setdefault(
            'periodic_analysis', {}
        )

        for key in (
            'branches',
            'locations',
            'attempts',
            'hopf',
        ):

            payload.setdefault(key, [])

        payload.setdefault('coverage', {})

        payload['status'] = 'failed'
        payload['reason'] = str(exc)

        payload['coverage'][
            'limit_reached'
        ] = True

        data.setdefault(
            'oscillations', []
        )

        data.setdefault(
            'stable_orbits', []
        )

        for branch in payload['branches']:

            for arm in branch['arms']:

                if arm['termination'] == 'running':
                    arm['termination'] = 'failed'

        try:

            (
                folder
                / f'periodic_error_H{depth:g}.txt'
            ).write_text(
                error_text,
                encoding='utf-8',
            )

            if dimensions == 1:

                write_json(
                    folder/f'periodic_H{depth:g}.json',
                    payload,
                )

                plot_stable_attractors(
                    data,
                    axes[0],
                    folder/'figures',
                )

            else:

                from model.periodic_orbits import (
                    export_periodic,
                )

                export_periodic(
                    data, folder, depth
                )

        except Exception:

            traceback.print_exc()

        periodic_checkpoint(
            'Periodic analysis failed · '
            'retaining equilibrium results'
        )

    # ----------------------------------------------------------
    # Save measured performance
    # ----------------------------------------------------------

    stats['oscillation_seconds'] = (
        time.monotonic()-oscillation_start
    )

    stats['elapsed_seconds'] = (
        time.monotonic()-started
    )

    if dimensions == 2:

        stats.update(
            discovery_seconds=discovery_seconds,
            refinement_seconds=refinement_seconds,
        )

    data['performance'] = stats

    write_json(
        folder/f'performance_H{depth:g}.json',
        stats,
    )

    report(
        1.,
        f'H={depth:g} m · complete',
    )

    return data


# ==============================================================
# Parallel depth calculations
# ==============================================================

def _child(
    p, axes, options, dimensions,
    folder, work, index,
):

    last = 0.

    def report(progress, phase):

        nonlocal last

        now = time.monotonic()

        if now-last >= .5 or progress == 1.:

            _atomic_bytes(
                work/f'{index}.progress',
                json.dumps(
                    [progress, phase]
                ).encode('utf-8'),
            )

            last = now

    try:

        data = _run_depth(
            p,
            axes,
            options,
            dimensions,
            folder,
            report,
        )

        _atomic_bytes(
            work/f'{index}.result',
            pickle.dumps(
                data,
                protocol=pickle.HIGHEST_PROTOCOL,
            ),
        )

    except BaseException:

        _atomic_bytes(
            work/f'{index}.error',
            traceback.format_exc().encode('utf-8'),
        )

        raise


def _parallel_depths(
    base, depths, axes, options,
    dimensions, ctx, workers,
):

    mpctx = mp.get_context('spawn')

    active = {}
    progress = [0.]*len(depths)
    results = {}

    next_index = 0

    with tempfile.TemporaryDirectory(
        prefix='simp_workers_',
        dir=ctx.folder,
    ) as temp:

        work = Path(temp)

        try:

            while len(results) < len(depths):

                while (
                    next_index < len(depths)
                    and len(active) < workers
                ):

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

                    old = {
                        k: os.environ.get(k)
                        for k in keys
                    }

                    try:

                        os.environ.update({
                            k: '1' for k in keys
                        })

                        process.start()

                    finally:

                        for k, v in old.items():

                            if v is None:

                                os.environ.pop(
                                    k, None
                                )

                            else:

                                os.environ[k] = v

                    active[i] = process

                phases = []

                for i, process in list(
                    active.items()
                ):

                    path = work/f'{i}.progress'

                    if path.exists():

                        try:

                            fraction, phase = json.loads(
                                path.read_text(
                                    encoding='utf-8'
                                )
                            )

                            progress[i] = max(
                                progress[i],
                                fraction,
                            )

                            phases.append(phase)

                        except (OSError, ValueError):

                            pass

                    if not process.is_alive():

                        process.join()

                        error = work/f'{i}.error'
                        result = work/f'{i}.result'

                        if (
                            process.exitcode != 0
                            or not result.exists()
                        ):

                            detail = (
                                error.read_text(
                                    encoding='utf-8'
                                )
                                if error.exists()
                                else f'exit code {process.exitcode}'
                            )

                            raise RuntimeError(
                                f'Depth H={depths[i]:g} '
                                f'worker failed: {detail}'
                            )

                        results[i] = pickle.loads(
                            result.read_bytes()
                        )

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

    return [
        results[i]
        for i in range(len(depths))
    ]


# ==============================================================
# Main continuation workflow
# ==============================================================

def run(values, ctx, dimensions):

    ctx.plot_start = 90.

    ctx.checkpoint(
        0,
        'Preparing continuation',
        force=True,
    )

    base, depths, axes, options = parse_job(
        values, dimensions
    )

    started = time.monotonic()

    workers = (
        min(
            len(depths),
            int(options.get('depth_workers', 1)),
        )
        if dimensions == 2 else 1
    )

    if workers > 1:

        datasets = _parallel_depths(
            base,
            depths,
            axes,
            options,
            dimensions,
            ctx,
            workers,
        )

    else:

        datasets = []

        for index, depth in enumerate(depths):

            def report(fraction, phase):

                ctx.checkpoint(
                    5+80*(index+fraction)/len(depths),
                    phase,
                )

            datasets.append(_run_depth(
                replace(base, H=depth),
                axes,
                options,
                dimensions,
                ctx.folder,
                report,
            ))

    from scripts.continuation_plots import (
        plot_1d,
        plot_2d,
    )

    ctx.checkpoint(
        90,
        'Rendering PNG and PDF figures',
        force=True,
    )

    if dimensions == 1:

        plot_1d(
            datasets,
            axes[0],
            ctx.folder/'figures',
        )

    else:

        plot_2d(
            datasets,
            axes,
            ctx.folder/'figures',
        )

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

        summary['refinement'] = {
            str(d['H']): d['refinement']
            for d in datasets
        }

        summary['unresolved_area'] = {
            str(d['H']): sum(
                r['polygon'].area
                for r in d['regions']
                if r['label'] == '?'
            )
            for d in datasets
        }

    write_json(
        ctx.folder/'summary.json',
        summary,
    )

    ctx.checkpoint(
        100,
        'Completed native continuation',
        force=True,
    )

    return datasets


def main(dimensions):

    mp.freeze_support()

    values, ctx, job = load_args()

    try:

        run(values, ctx, dimensions)

    except StopRequested:

        return 130

    return 0
