
"""Shared CLI runner for native 1D and 2D continuation jobs."""

from dataclasses import replace
from pathlib import Path

import csv
import json
import time
import re
import os
import multiprocessing as mp
import traceback
import pickle
import tempfile

from app.core.control import load_args, StopRequested
from model.continuation_config import parse_job
from model.periodic_config import PERIODIC_DEPTH_BUDGET


def write_csv(path, rows):
    if not rows:
        return

    fields = list(dict.fromkeys(
        key for row in rows for key in row
    ))

    with Path(path).open(
        'w', newline='', encoding='utf-8'
    ) as stream:

        writer = csv.DictWriter(
            stream, fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)


def _run_depth(
    p, axes, options, dimensions, folder, report
):
    """Calculate equilibria, bifurcations and periodic orbits."""

    from model.pyoomph_continuation import (
        continue_1d,
        continue_2d,
        locate_events,
        native_session,
    )

    from model.continuation_refinement import refine_regions

    depth = p.H
    started = time.monotonic()

    local_progress = 0.
    current_start = ''

    def checkpoint(message):
        """Report equilibrium and bifurcation progress."""

        nonlocal local_progress, current_start

        counts = re.search(r'(\d+)/(\d+)', message)

        if counts and 'Equilibrium overview' in message:
            local_progress = max(
                local_progress,
                .15 * int(counts[1]) / int(counts[2]),
            )

        elif counts and 'Checking start' in message:
            local_progress = max(
                local_progress,
                .15 + .60 * int(counts[1]) / int(counts[2]),
            )

            current_start = counts[0]

        elif counts and 'Checking region' in message:
            local_progress = max(
                local_progress,
                .75 + .20 * int(counts[1]) / int(counts[2]),
            )

        elif 'correction' in message:
            local_progress = max(
                local_progress,
                .76 if dimensions == 1 else .15,
            )

        elif 'Native 2D' in message:
            local_progress = max(local_progress, .45)

        phase = f'H={depth:g} m · {message}'

        if (
            current_start
            and message.startswith('Native ')
            and dimensions == 1
        ):
            phase += f' · start {current_start}'

        # Reserve the final 25% of each depth for periodic analysis.
        report(.75 * local_progress, phase)

    # ============================================================
    # STAGE 1: Equilibria and native bifurcation continuation
    # ============================================================

    with native_session(
        reuse=(
            dimensions == 2
            and str(options.get('reuse_native', 'off')) == 'on'
        ),
        step_factor=(
            float(options.get('adaptive_step_factor', 1))
            if dimensions == 2 else 1
        ),
    ) as stats:

        common = dict(
            discovery_points=int(options['discovery_points']),
            max_steps=int(options['max_steps']),
            checkpoint=checkpoint,
        )

        if dimensions == 1:

            data = continue_1d(
                p, axes[0], **common
            )

            if (
                any(
                    row['community'] != '0'
                    for row in data['scan']
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
                    'Inspect the worker log; discovery points '
                    'are not a continuation result.'
                )

            events = []

            # Detect events along the actual native branches.
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

                    duplicate = any(
                        old['kind'] == event['kind']
                        and old['community'] == event['community']
                        and old['invader'] == event['invader']
                        and abs(old['u'] - event['u']) < 1e-5
                        for old in events
                    )

                    if not duplicate:
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
                folder / f'branches_H{depth:g}.csv',
                rows,
            )

        else:

            discovery_start = time.monotonic()

            data = continue_2d(
                p,
                axes,
                transects=int(options['transects']),
                discovery_refinement=int(
                    options['discovery_refinement']
                ),
                **common,
            )

            discovery_seconds = (
                time.monotonic() - discovery_start
            )

            refinement_start = time.monotonic()

            def save_refinement_checkpoint(current, history):
                """Preserve partial native event curves."""

                rows = [
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
                ]

                write_csv(
                    folder / f'curves_H{depth:g}.csv',
                    rows,
                )

                (
                    folder
                    / f'refinement_checkpoint_H{depth:g}.json'
                ).write_text(
                    json.dumps(
                        dict(
                            rounds=history,
                            curve_count=len(current['curves']),
                            status='partial_checkpoint',
                        ),
                        indent=2,
                    ),
                    encoding='utf-8',
                )

            save_refinement_checkpoint(data, [])

            data = refine_regions(
                data,
                p,
                axes,
                int(options['region_samples']),
                int(options['max_steps']),
                checkpoint,
                max_rounds=int(options['refinement_rounds']),
                searches_per_round=int(
                    options['refinement_searches']
                ),
                on_round=save_refinement_checkpoint,
            )

            refinement_seconds = (
                time.monotonic() - refinement_start
            )

            (
                folder / f'geometry_H{depth:g}.json'
            ).write_text(
                json.dumps(
                    data.get('geometry_diagnostics', []),
                    indent=2,
                ),
                encoding='utf-8',
            )

            (
                folder / f'refinement_H{depth:g}.json'
            ).write_text(
                json.dumps(data['refinement'], indent=2),
                encoding='utf-8',
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
                folder / f'curves_H{depth:g}.csv',
                rows,
            )

            write_csv(
                folder / f'region_samples_H{depth:g}.csv',
                [
                    dict(
                        sample['selected'] or {},
                        region=i,
                        label=region['label'],
                        u=sample['u'],
                        v=sample['v'],
                        signature=sample['signature'],
                    )
                    for i, region in enumerate(
                        data['regions'], 1
                    )
                    for sample in region['samples']
                ],
            )

            (
                folder / f'regions_H{depth:g}.json'
            ).write_text(
                json.dumps(
                    [
                        dict(
                            label=region['label'],
                            normalized_polygon_wkt=(
                                region['polygon'].wkt
                            ),
                            area=region['polygon'].area,
                            signatures=region['signatures'],
                            reasons=region['reasons'],
                        )
                        for region in data['regions']
                    ],
                    indent=2,
                ),
                encoding='utf-8',
            )

        # Save equilibrium results before any periodic calculations.
        write_csv(
            folder / f'overview_H{depth:g}.csv',
            data['scan'],
        )

        write_csv(
            folder / f'events_H{depth:g}.csv',
            data['events'],
        )

        (
            folder / f'diagnostics_H{depth:g}.json'
        ).write_text(
            json.dumps(data['diagnostics'], indent=2),
            encoding='utf-8',
        )

    # The native equilibrium session has now closed.
    # No cached equilibrium problem remains active here.

    stats['equilibrium_seconds'] = (
        time.monotonic() - started
    )

    # ============================================================
    # STAGE 2: Periodic-orbit discovery and continuation
    # ============================================================

    from model.periodic_orbits import (
        analyze_periodic,
        export_periodic,
    )

    oscillation_start = time.monotonic()

    def periodic_checkpoint(message):
        """Report approximate progress within the periodic budget."""

        elapsed = time.monotonic() - oscillation_start

        fraction = min(
            elapsed / PERIODIC_DEPTH_BUDGET,
            .99,
        )

        report(
            .75 + .24 * fraction,
            f'H={depth:g} m · {message}',
        )

    periodic_checkpoint('Starting periodic analysis')

    try:
        analyze_periodic(
            data,
            p,
            axes,
            options,
            periodic_checkpoint,
            folder,
        )

    except StopRequested:
        # Preserve the existing cooperative stop behaviour.
        raise

    except Exception as exc:
        # Do not discard successfully computed equilibria because
        # an optional numerical stage encountered a Python error.
        # Periodic analysis is still attempted for every run.
        traceback.print_exc()

        payload = data.setdefault(
            'periodic_analysis', {}
        )

        payload.setdefault('branches', [])
        payload.setdefault('locations', [])
        payload.setdefault('attempts', [])
        payload.setdefault('hopf', [])
        payload.setdefault('coverage', {})

        payload['status'] = 'failed'
        payload['reason'] = str(exc)
        payload['coverage']['limit_reached'] = True

        data.setdefault('oscillations', [])

        for branch in payload['branches']:
            for arm in branch['arms']:
                if arm['termination'] == 'running':
                    arm['termination'] = 'failed'

        error_text = traceback.format_exc()

        try:
            (
                folder / f'periodic_error_H{depth:g}.txt'
            ).write_text(
                error_text,
                encoding='utf-8',
            )

            export_periodic(data, folder, depth)

        except Exception:
            traceback.print_exc()

        periodic_checkpoint(
            'Periodic analysis failed '
            '· retaining equilibrium results'
        )

    stats['oscillation_seconds'] = (
        time.monotonic() - oscillation_start
    )

    stats['elapsed_seconds'] = (
        time.monotonic() - started
    )

    if dimensions == 2:
        stats.update(
            discovery_seconds=discovery_seconds,
            refinement_seconds=refinement_seconds,
        )

    data['performance'] = stats

    (
        folder / f'performance_H{depth:g}.json'
    ).write_text(
        json.dumps(stats, indent=2),
        encoding='utf-8',
    )

    report(1., f'H={depth:g} m · complete')

    return data


def _atomic_bytes(path, data):
    """Write private multiprocessing data atomically."""

    temp = path.with_suffix('.tmp')
    temp.write_bytes(data)

    for attempt in range(8):
        try:
            os.replace(temp, path)
            return

        except PermissionError:
            if attempt == 7:
                raise

            time.sleep(.025 * (attempt + 1))


def _child(
    p, axes, options, dimensions, folder, work, index
):
    """Independent spawned depth worker."""

    last = 0.

    def report(progress, phase):
        nonlocal last

        now = time.monotonic()

        if now - last >= .5 or progress == 1.:

            _atomic_bytes(
                work / f'{index}.progress',
                json.dumps(
                    [progress, phase]
                ).encode('utf-8'),
            )

            last = now

    try:
        data = _run_depth(
            p, axes, options, dimensions, folder, report
        )

        _atomic_bytes(
            work / f'{index}.result',
            pickle.dumps(
                data,
                protocol=pickle.HIGHEST_PROTOCOL,
            ),
        )

    except BaseException:

        _atomic_bytes(
            work / f'{index}.error',
            traceback.format_exc().encode('utf-8'),
        )

        raise


def _parallel_depths(
    base, depths, axes, options, dimensions, ctx, workers
):
    """Run independent depth calculations in spawned processes."""

    mpctx = mp.get_context('spawn')

    active = {}
    progress = [0.] * len(depths)
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
                            replace(base, H=depths[i]),
                            axes,
                            options,
                            dimensions,
                            ctx.folder,
                            work,
                            i,
                        ),
                    )

                    # Prevent multiplication of native BLAS threads.
                    keys = (
                        'OMP_NUM_THREADS',
                        'OPENBLAS_NUM_THREADS',
                        'MKL_NUM_THREADS',
                        'BLIS_NUM_THREADS',
                        'NUMEXPR_NUM_THREADS',
                        'PYTHONFAULTHANDLER',
                    )

                    old = {
                        key: os.environ.get(key)
                        for key in keys
                    }

                    try:
                        os.environ.update({
                            key: '1' for key in keys
                        })

                        process.start()

                    finally:
                        for key, value in old.items():

                            if value is None:
                                os.environ.pop(key, None)

                            else:
                                os.environ[key] = value

                    active[i] = process

                phases = []

                for i, process in list(active.items()):

                    path = work / f'{i}.progress'

                    if path.exists():
                        try:
                            fraction, phase = json.loads(
                                path.read_text(
                                    encoding='utf-8'
                                )
                            )

                            progress[i] = max(
                                progress[i], fraction
                            )

                            phases.append(phase)

                        except (OSError, ValueError):
                            pass

                    if not process.is_alive():

                        process.join()

                        error = work / f'{i}.error'
                        result = work / f'{i}.result'

                        if (
                            process.exitcode != 0
                            or not result.exists()
                        ):
                            detail = (
                                error.read_text(
                                    encoding='utf-8'
                                )
                                if error.exists()
                                else (
                                    f'exit code '
                                    f'{process.exitcode}'
                                )
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
                    5 + 80 * sum(progress) / len(depths),
                    (
                        f'Parallel depths '
                        f'{len(results)}/{len(depths)} · '
                        + ' | '.join(phases[:workers])
                    ),
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
        results[i] for i in range(len(depths))
    ]


def run(values, ctx, dimensions):
    ctx.plot_start = 90.

    ctx.checkpoint(
        0, 'Preparing continuation', force=True
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

    # ============================================================
    # Compute the individual depths
    # ============================================================

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
                    5 + 80 * (index + fraction) / len(depths),
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

    # ============================================================
    # Generate figures from all available verified results
    # ============================================================

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
            ctx.folder / 'figures',
        )

    else:
        plot_2d(
            datasets,
            axes,
            ctx.folder / 'figures',
        )

    summary = dict(
        engine='pyoomph native pseudo-arclength',
        dimensions=dimensions,
        depths=depths,
        axes=[vars(axis) for axis in axes],
        seconds=time.monotonic() - started,
        equation_source='model/shared_eqn.py',

        limitations=(
            'Finite periodic discovery cannot prove absence '
            'of cycles. Only verified periodic BVP solutions '
            'are shown. Failed or budget-limited searches '
            'remain unresolved. Two-dimensional periodic '
            'results represent checked locations and '
            'continuation transects, not certified regions.'
        ),

        diagnostics=sum(
            len(data['diagnostics'])
            for data in datasets
        ),

        depth_workers=workers,

        performance={
            str(data['H']): data['performance']
            for data in datasets
        },
    )

    summary['periodic'] = {
        str(data['H']): dict(
            verified_points=len(
                data.get('oscillations', [])
            ),
            status=data.get(
                'periodic_analysis', {}
            ).get('status', 'unavailable'),
            coverage=data.get(
                'periodic_analysis', {}
            ).get('coverage', {}),
        )
        for data in datasets
    }

    if dimensions == 2:

        summary['refinement'] = {
            str(data['H']): data['refinement']
            for data in datasets
        }

        summary['unresolved_area'] = {
            str(data['H']): sum(
                region['polygon'].area
                for region in data['regions']
                if region['label'] == '?'
            )
            for data in datasets
        }

    (
        ctx.folder / 'summary.json'
    ).write_text(
        json.dumps(summary, indent=2),
        encoding='utf-8',
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
