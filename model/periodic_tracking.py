"""Adaptive native periodic-orbit continuation."""

import time
import numpy as np

from model import periodic_native as native
from model.periodic_math import orbit_signature, same_orbit
from model.periodic_config import node_count


def needs_verification(row, previous, steps, resolution):
    """Determine whether a fully verified output point is needed."""
    if previous is None:
        return True

    if steps >= 8:
        return True

    if row['stability'] != previous['stability']:
        return True

    if row['stability'] == 'unresolved':
        return True

    # Resolve the neighborhood of a bifurcation more densely.
    if abs(row.get('spectral_radius', 1.)-1.) < .02:
        return True

    parameter_gap = max(abs(row[k]-previous[k]) for k in 'uv' if k in row)
    if parameter_gap >= min(.0025, .25/resolution):
        return True

    # Compare the complete biomass envelope and period.
    change = np.max(abs(orbit_signature(row)-orbit_signature(previous)))
    return bool(change >= .025)


def hopf_endpoint(problem, orbit, task, previous_amplitude):
    """Check whether a shrinking cycle approaches a classified Hopf."""
    if previous_amplitude is None:
        return None

    samples = native.sample_orbit(problem, orbit, 256)
    if not np.all(np.isfinite(samples)) or np.min(samples) <= 0:
        return None

    mean = np.mean(samples, axis=0)
    amplitude = float(np.max(np.ptp(samples, axis=0)/np.maximum(mean, 1e-8)))
    if amplitude > .07 or amplitude >= previous_amplitude:
        return None

    uv = np.array([float(c.value) for c in problem.controls])
    period = float(orbit.get_T(dimensional=False))
    for event in task.get('hopf_points', []):
        if event.get('classification_status') != 'verified':
            continue

        if not event.get('classification', '').startswith('hopf_'):
            continue

        target = np.array([float(event[k]) for k in 'uv'[:len(uv)]])
        gap = float(np.max(abs(uv-target)))
        if gap > 1e-4:
            continue

        state = np.array([event[c] for c in 'MSP'], dtype=float)
        state_error = float(np.max(abs(mean-state)/np.maximum(abs(state), 1e-8)))
        if state_error > .03:
            continue

        expected_period = float(event.get('hopf_period', 0.))
        if expected_period <= 0:
            continue

        period_error = abs(period-expected_period)/expected_period
        if period_error > .05:
            continue

        return dict(
            hopf_coordinates=target.tolist(),
            endpoint_coordinates=uv.tolist(),
            parameter_gap=gap,
            relative_amplitude=amplitude,
            state_error=state_error,
            period_error=period_error,
        )

    return None


def trace(task, base, axes, options, emit):
    """Trace using a persistent mesh and adaptive output verification."""
    seed = task['seed']
    axis = task['axis']
    direction = task['direction']
    initial_level = int(seed.get('mesh_level', 0))

    # Retain a useful working mesh, rather than restarting from the
    # original 30-node discretization at every continuation point.
    working_level = max(0, initial_level-2)
    preferred_level = max(working_level+1, initial_level)
    logs = native.resample_logs(seed['log_states'], node_count(options, working_level),)
    uv = seed['uv']
    started = time.monotonic()
    stats = dict(
        internal_steps=0,
        verified_points=0,
        working_level=working_level,
        working_nodes=node_count(options, working_level),
        continuation_seconds=0.,
        inspection_seconds=0.,
        verification_seconds=0.,
    )

    def statistics():
        return dict(stats, elapsed_seconds=time.monotonic()-started,)

    def terminate(reason, **details):
        emit(dict(type='termination', reason=reason, statistics=statistics(), **details,))

    with native.periodic_problem(base, axes, uv, np.exp(logs[0])) as problem:
        orbit = native.install_guess(problem, logs, seed['period'], options)
        initial = native.inspect_orbit(problem, orbit, base, axes, options)
        previous = initial
        previous_verified = None
        previous_delta = None
        previous_amplitude = initial['relative_amplitude']
        steps_since_verification = 0
        turns = 0
        maximum_excursion = 0.
        nominal = min(options['periodic_max_step'], 1/axes[axis].resolution,)

        # Increase the permitted native arclength step in smooth
        # regions, while retaining a smaller cap near events.
        cap = nominal
        maximum_cap = min(.06, 4*nominal)
        ds = direction*min(nominal/4, options['periodic_hopf_offset']/4,)
        problem.reset_arc_length_parameters()
        coordinate = 'uv'[axis]
        for step in range(options['periodic_max_steps']):
            t = time.monotonic()
            ds = problem.arclength_continuation(
                problem.controls[axis],
                ds,
                max_ds=cap,
                min_ds=options['periodic_min_step'],
                **native.solve_options(options),
            )
            stats['continuation_seconds'] += time.monotonic()-t
            stats['internal_steps'] += 1
            current_uv = [float(c.value) for c in problem.controls]
            if any(not 0 <= u <= 1 for u in current_uv):
                terminate(
                    'range_boundary',
                    last_coordinates=current_uv,
                    unverified_internal_steps=steps_since_verification,
                )
                return

            t = time.monotonic()
            try:
                row = native.inspect_orbit(problem, orbit, base, axes, options)

            except ValueError as exc:
                if 'Collapsed stationary solution' not in str(exc):
                    raise

                endpoint = hopf_endpoint(problem, orbit, task, previous_amplitude)
                if (
                    endpoint is not None
                    and previous_verified is not None
                    and previous_verified['relative_amplitude'] < .15
                ):
                    terminate('approached_verified_hopf', **endpoint,)
                    return

                raise

            stats['inspection_seconds'] += time.monotonic()-t
            steps_since_verification += 1
            delta = row[coordinate]-previous[coordinate]
            turning = (
                previous_delta is not None
                and abs(previous_delta) > 1e-10
                and abs(delta) > 1e-10
                and delta*previous_delta < 0
            )
            if turning:
                turns += 1

            stability_change = (row['stability'] != previous['stability'])
            endpoint = None
            if row['relative_amplitude'] < .08:
                endpoint = hopf_endpoint(problem, orbit, task, previous_amplitude)

            verify = (
                turning
                or stability_change
                or endpoint is not None
                or needs_verification(
                    row,
                    previous_verified,
                    steps_since_verification,
                    axes[axis].resolution,
                )
                or step == options['periodic_max_steps']-1
            )
            if verify:
                t = time.monotonic()
                saved_row, saved = native.verify_point(
                    problem,
                    orbit,
                    base,
                    axes,
                    options,
                    working_level=working_level,
                    preferred_level=preferred_level,
                    coarse=row,
                )
                stats['verification_seconds'] += (time.monotonic()-t)
                preferred_level = saved_row['verification_level']
                stats['verified_points'] += 1
                saved_row.update(
                    step=step+1,
                    direction=direction,
                    continuation_axis=axis,
                    internal_steps_since_previous=(
                        steps_since_verification
                    ),
                    tracking_statistics=statistics(),
                )
                if turning:
                    saved_row['event'] = 'cycle_fold_bracket'

                if stability_change:
                    saved_row['stability_transition'] = True

                emit(dict(type='orbit', row=saved_row, seed=saved,))
                previous_verified = saved_row
                steps_since_verification = 0
                if endpoint is not None:
                    terminate('approached_verified_hopf', **endpoint,)
                    return

                # Avoid mistaking slowly advancing solutions for
                # closed periodic branches in parameter space.
                distance = float(np.max(abs(orbit_signature(saved_row) - orbit_signature(initial))))
                maximum_excursion = max(maximum_excursion, distance)
                if (
                    turns >= 2
                    and maximum_excursion > .05
                    and same_orbit(saved_row, initial, .002)
                    and max(
                        abs(saved_row[k]-initial[k])
                        for k in 'uv'[:len(axes)]
                    ) < 1e-4
                ):
                    terminate('closed_loop')
                    return

            else:

                # A successful intermediate correction is progress,
                # but is not a fully mesh-verified output point.
                emit(dict(
                    type='trace_progress',
                    coordinates=current_uv,
                    amplitude=row['relative_amplitude'],
                    statistics=statistics(),
                ))

            # Adapt the arclength cap using actual solution changes.
            shape_change = float(np.max(abs(orbit_signature(row)-orbit_signature(previous))))
            critical = (
                turning
                or stability_change
                or abs(row.get('spectral_radius', 1.)-1.) < .02
                or row['relative_amplitude'] < .1
            )
            if critical:
                cap = nominal
                ds = np.sign(ds)*min(abs(ds), nominal/4)

            elif shape_change < .015 and abs(delta) < .001:
                cap = min(maximum_cap, 1.25*cap)

            elif shape_change > .05 or abs(delta) > .005:
                cap = max(nominal/4, .5*cap)
                ds = np.sign(ds)*min(abs(ds), cap)

            previous_delta = delta
            previous_amplitude = row['relative_amplitude']
            previous = row

        terminate('step_limit')
