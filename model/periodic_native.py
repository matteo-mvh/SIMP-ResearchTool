
"""Native periodic BVPs, executed inside disposable worker processes."""

from contextlib import contextmanager
from tempfile import TemporaryDirectory

import numpy as np

from model.pyoomph_continuation import NativeProblem, parameters_at
from model.shared_eqn import solve_equilibria
from model.periodic_config import node_count
from model.periodic_math import (
    admissible_msp,
    sample_metrics,
    floquet_classification,
    spectrum_distance,
    same_orbit,
)


@contextmanager
def periodic_problem(base, axes, uv, state, kind='', solved_axis=0):
    """Initialize an independent native periodic problem."""

    with TemporaryDirectory(prefix='simp_periodic_') as folder:
        with NativeProblem(
            base, axes, uv, state, 'MSP',
            kind=kind, solved_axis=solved_axis,
        ) as problem:

            problem.set_output_directory(folder)

            if kind == 'hopf':
                problem.correct()
            else:
                problem.setup_for_stability_analysis(
                    analytic_hessian=True
                )
                problem.initialise()

            try:
                yield problem
            finally:
                problem.deactivate_bifurcation_tracking()


def solve_options(options):
    return dict(
        newton_solver_tolerance=options['periodic_newton_tolerance'],
        max_newton_iterations=options['periodic_newton_iterations'],
        globally_convergent_newton=(
            options['periodic_global_newton'] == 'on'
        ),
    )


def install_guess(problem, log_states, period, options):
    """Install an approximate periodic trajectory and solve its BVP."""

    problem.deactivate_bifurcation_tracking()

    dofs = []
    ode = problem.get_ode('ecosystem')

    for state in log_states:
        ode.set_value(**dict(zip('MSP', map(float, state))))
        dofs.append(problem.get_current_dofs()[0].copy())

    problem.set_current_dofs(dofs[0])

    orbit = problem.activate_periodic_orbit_handler(
        period,
        history_dofs=dofs[1:],
        mode=options['periodic_mode'],
        order=options['periodic_order'],
        T_constraint='phase',
    )

    problem.solve(**solve_options(options))

    return orbit


def sample_orbit(problem, orbit, count):
    """Sample an already-computed periodic solution."""

    states = []

    for _ in orbit.iterate_over_samples(N=count, endpoint=False):
        states.append(problem.vector()[0])

    return np.asarray(states)


def snapshot(problem, orbit, count):
    """Save a reusable periodic solution."""

    return dict(
        log_states=np.log(
            sample_orbit(problem, orbit, count)
        ).tolist(),
        period=float(orbit.get_T(dimensional=False)),
        uv=[float(c.value) for c in problem.controls],
    )


def resample_logs(log_states, count):
    """Periodic interpolation onto a different temporal mesh."""

    values = np.asarray(log_states)
    phase = np.arange(len(values)+1)/len(values)

    return np.column_stack([
        np.interp(
            np.arange(count)/count,
            phase,
            np.r_[column, column[0]],
        )
        for column in values.T
    ])


def inspect_orbit(problem, orbit, base, axes, options):
    """Check physical admissibility, residual and Floquet spectrum."""

    uv = [float(c.value) for c in problem.controls]

    if any(not 0 <= v <= 1 for v in uv):
        raise ValueError('Parameter boundary reached')

    p = parameters_at(base, axes, uv)

    count = max(512, 8*orbit.get_num_time_steps())

    row = sample_metrics(
        sample_orbit(problem, orbit, count),
        float(orbit.get_T(dimensional=False)),
        p,
        options,
    )

    residual = float(np.max(abs(problem.get_residuals())))

    if (
        not np.isfinite(residual)
        or residual > max(
            10*options['periodic_newton_tolerance'], 1e-10
        )
    ):
        raise ValueError(
            f'Periodic residual {residual:g} exceeds tolerance'
        )

    multipliers = orbit.get_floquet_multipliers(
        ignore_periodic_unity=False,
        valid_threshold=None,
        method='condensed',
        sampling_order=options['periodic_order'],
    )

    if not np.all(np.isfinite(multipliers)):
        raise ValueError('Nonfinite Floquet multipliers')

    row.update(floquet_classification(
        multipliers,
        options['periodic_floquet_tolerance'],
    ))

    row.update(
        residual=residual,
        nodes=orbit.get_num_time_steps(),
        community='MSP',
        H=base.H,
        **{key: float(v) for key, v in zip('uv', uv)},
        **{
            key: float(a.physical(v))
            for key, a, v in zip('xy', axes, uv)
        },
    )

    return row


def equilibrium_context(p):
    """Find coexisting equilibria without rejecting valid orbits."""

    try:
        roots = solve_equilibria(p)

        stable = '+'.join(sorted({
            e.community for e in roots
            if e.stability.classification == 'stable'
        }))

        return dict(
            stable_equilibria=stable,
            equilibrium_search_status=(
                'msp_found'
                if any(e.community == 'MSP' for e in roots)
                else 'msp_unresolved'
            ),
        )

    except (
        ValueError,
        RuntimeError,
        ArithmeticError,
        np.linalg.LinAlgError,
    ) as exc:
        return dict(
            stable_equilibria='',
            equilibrium_search_status='failed',
            equilibrium_search_error=str(exc),
        )


def verify_point(problem, orbit, base, axes, options,
                 working_level=0, preferred_level=None, coarse=None):
    """Verify a periodic solution through independent finer-mesh solves."""

    if coarse is None:
        coarse = inspect_orbit(
            problem, orbit, base, axes, options
        )

    uv = [coarse[k] for k in 'uv'[:len(axes)]]
    p = parameters_at(base, axes, uv)

    # Equilibria provide information about coexistence with cycles.
    # Their numerical discovery is not a condition for BVP validity.
    context = equilibrium_context(p)

    previous = coarse
    previous_level = working_level

    seed = snapshot(
        problem, orbit, node_count(options, working_level)*2
    )

    maximum = options['periodic_refinement_levels']

    # Start near the previously successful verification mesh.
    # Still require comparison of two successive mesh levels.
    start = max(
        working_level+1,
        (preferred_level or 1)-1,
    )

    start = min(start, maximum)

    for level in range(start, maximum+1):

        n = node_count(options, level)

        logs = resample_logs(
            seed['log_states'], n
        )

        with periodic_problem(
            base, axes, uv, np.exp(logs[0])
        ) as fine:

            refined = install_guess(
                fine, logs, seed['period'], options
            )

            row = inspect_orbit(
                fine, refined, base, axes, options
            )

            keys = ['period'] + [
                component+'_'+bound
                for component in ('M', 'S', 'P_area', 'N_area')
                for bound in ('min', 'max', 'mean')
            ]

            error = max(
                abs(row[k]-previous[k])/max(abs(row[k]), 1e-5)
                for k in keys
            )

            floquet_error = spectrum_distance(
                previous['floquet'], row['floquet']
            )

            row.update(
                mesh_error=float(error),
                floquet_mesh_error=floquet_error,
                verification_level=level,
                **context,
            )

            saved = snapshot(fine, refined, n)
            saved['mesh_level'] = level

            if (
                level == previous_level+1
                and error <= options['periodic_validation_tolerance']
                and floquet_error <= options['periodic_validation_tolerance']
                and row['balance_defect'] <= options['periodic_defect_tolerance']
                and row.get('trivial_error', float('inf'))
                    <= options['periodic_floquet_tolerance']
            ):

                if row['stability'] != previous['stability']:
                    row['stability'] = 'unresolved'

                elif abs(row.get('spectral_radius', 1)-1) <= max(
                    options['periodic_floquet_tolerance'],
                    2*floquet_error,
                ):
                    row['stability'] = 'nonhyperbolic'

                row['status'] = 'verified_periodic'

                return row, saved

            previous = row
            previous_level = level
            seed = saved

    raise ValueError(
        'Periodic orbit failed mesh convergence or balance verification: '
        + str({
            k: previous.get(k)
            for k in (
                'mesh_error',
                'floquet_mesh_error',
                'balance_defect',
                'trivial_error',
                'stability',
            )
        })
    )


def discover_task(task, base, axes, options, emit):
    """Initialize an orbit from Hopf, reuse or deterministic guesses."""

    uv = task['uv']
    p = parameters_at(base, axes, uv)

    if task['kind'] == 'reuse':

        # Reuse does not require rediscovering an equilibrium.
        logs = resample_logs(
            task['seed']['log_states'],
            node_count(options),
        )

        state = np.exp(logs[0])
        kind = ''

    else:
        row = task['equilibrium']

        if not admissible_msp(row, p):
            raise ValueError(
                'Initial guess requires an admissible MSP equilibrium'
            )

        state = np.array([row[c] for c in 'MSP'])

        kind = (
            'hopf' if task['kind'] == 'hopf'
            else ''
        )

    with periodic_problem(
        base, axes, uv, state, kind,
        task.get('axis', 0),
    ) as problem:

        info = {}

        if kind == 'hopf':

            orbit = problem.switch_to_hopf_orbit(
                dparam=options['periodic_hopf_offset'],
                NT=node_count(options),
                mode=options['periodic_mode'],
                order=options['periodic_order'],
                solve_kwargs=solve_options(options),
            )

            coefficient = float(
                np.real(orbit.emerging_info['lyap_coeff'])
            )

            if not np.isfinite(coefficient):
                raise ValueError('Nonfinite Hopf Lyapunov coefficient')

            info = dict(
                lyapunov_coefficient=coefficient,
                hopf_classification=(
                    'supercritical' if coefficient < -1e-10
                    else 'subcritical' if coefficient > 1e-10
                    else 'degenerate_or_unresolved'
                ),
                hopf_uv=uv,
                hopf_equilibrium=state.tolist(),
            )

            emit(dict(type='hopf', **info))

        elif task['kind'] == 'reuse':

            orbit = install_guess(
                problem,
                logs,
                task['seed']['period'],
                options,
            )

        else:

            theta = (
                2*np.pi*np.arange(node_count(options))
                / node_count(options)
            )

            logs = (
                np.log(state)[None, :]
                + task['amplitude']*(
                    np.cos(theta)[:, None]*task['a']
                    + np.sin(theta)[:, None]*task['b']
                )
            )

            # Keep guesses within the nitrogen inventory.
            for _ in range(12):
                states = np.exp(logs)

                if np.max(
                    states[:, 0]+p.H*states[:, 2]
                ) < p.N_wc:
                    break

                logs = (
                    np.log(state)
                    + (logs-np.log(state))*.5
                )

            orbit = install_guess(
                problem, logs, task['period'], options
            )

        verified, seed = verify_point(
            problem, orbit, base, axes, options
        )

        verified.update(info)

        emit(dict(
            type='orbit',
            row=verified,
            seed=seed,
        ))


def trace_task(task, base, axes, options, emit):
    from model.periodic_tracking import trace
    return trace(task, base, axes, options, emit)


def execute_task(task, base, axes, options, emit):

    if task['kind'] == 'trace':
        trace_task(task, base, axes, options, emit)

    else:
        discover_task(task, base, axes, options, emit)
