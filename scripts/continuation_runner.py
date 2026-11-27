
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


def write_csv(path, rows):
    if not rows:
        return

    fields = list(dict.fromkeys(
        k for row in rows for k in row
    ))

    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_depth(p, axes, options, dimensions, folder, report):
    """Compute equilibrium branches, bifurcations and periodic solutions."""

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
        nonlocal local_progress, current_start

        counts = re.search(r'(\d+)/(\d+)', message)

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
            local_progress = max(local_progress, .45)

        phase = f'H={depth:g} m · {message}'

        if (
            current_start
            and message.startswith('Native ')
            and dimensions == 1
        ):
            phase += f' · start {current_start}'

        report(.75*local_progress, phase)

    # ------------------------------------------------------------
    # Equilibrium and bifurcation continuation.
    # ------------------------------------------------------------

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

            data = continue_1d(p, axes[0], **common)

            if (
                any(r['community'] != '0' for r in data['scan'])
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
                for branch, path in enumerate(data['paths'], 1)
                for i, row in enumerate(path)
            ]

            write_csv(folder/f'branches_H{depth:g}.csv', rows)

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
                time.monotonic()-discovery_start
            )

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

                (
                    folder/f'refinement_checkpoint_H{depth:g}.json'
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
                searches_per_round=int(options['refinement_searches']),
                on_round=save_refinement_checkpoint,
            )

            refinement_seconds = (
                time.monotonic()-refinement_start
            )

            (
                folder/f'geometry_H{depth:g}.json'
            ).write_text(
                json.dumps(
                    data.get('geometry_diagnostics', []),
                    indent=2,
                ),
                encoding='utf-8',
            )

            (
                folder/f'refinement_H{depth:g}.json'
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
                for curve, path in enumerate(data['curves'], 1)
                for i, row in enumerate(path)
            ]

            write_csv(folder/f'curves_H{depth:g}.csv', rows)

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
                    for i, r in enumerate(data['regions'], 1)
                    for s in r['samples']
                ],
            )

            (
                folder/f'regions_H{depth:g}.json'
            ).write_text(
                json.dumps(
                    [
                        dict(
                            label=r['label'],
                            normalized_polygon_wkt=r['polygon'].wkt,
                            area=r['polygon'].area,
                            signatures=r['signatures'],
                            reasons=r['reasons'],
                        )
                        for r in data['regions']
                    ],
                    indent=2,
                ),
                encoding='utf-8',
            )

        # Save equilibrium results before periodic analysis starts.
        write_csv(
            folder/f'overview_H{depth:g}.csv',
            data['scan'],
        )

        write_csv(
            folder/f'events_H{depth:g}.csv',
            data['events'],
        )

        (
            folder/f'diagnostics_H{depth:g}.json'
        ).write_text(
            json.dumps(data['diagnostics'], indent=2),
            encoding='utf-8',
        )

    stats['equilibrium_seconds'] = time.monotonic()-started

    # ------------------------------------------------------------
    # Periodic analysis.
    # All native equilibrium problems have been released.
    # ------------------------------------------------------------

    oscillation_start = time.monotonic()
    periodic_progress = .75

    def periodic_checkpoint(message):
        """Approximate progress based on completed search locations."""

        nonlocal periodic_progress

        if message.startswith('Periodic Hopf seeds'):

            counts = re.search(r'(\d+)/(\d+)', message)

            if counts:
                fraction = (
                    (int(counts[1])-1)/max(int(counts[2]), 1)
                )

                periodic_progress = max(
                    periodic_progress,
                    .75+.06*fraction,
                )

        elif message.startswith('Periodic discovery'):

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

    periodic_checkpoint('Starting automatic periodic analysis')

    try:

        from model.periodic_orbits import analyze_periodic

        analyze_periodic(
            data,
            p,
            axes,
            options,
            periodic_checkpoint,
            folder,
        )

    except StopRequested:
        raise

    except Exception as exc:

        # Periodic failures must not discard valid equilibrium results.
        traceback.print_exc()
        error_text = traceback.format_exc()

        payload = data.setdefault('periodic_analysis', {})

        for key in ('branches', 'locations', 'attempts', 'hopf'):
            payload.setdefault(key, [])

        payload.setdefault('coverage', {})
        payload['status'] = 'failed'
        payload['reason'] = str(exc)
        payload['coverage']['limit_reached'] = True

        data.setdefault('oscillations', [])

        for branch in payload['branches']:
            for arm in branch['arms']:
                if arm['termination'] == 'running':
                    arm['termination'] = 'failed'

        try:

            (
                folder/f'periodic_error_H{depth:g}.txt'
            ).write_text(
                error_text,
                encoding='utf-8',
            )

            from model.periodic_orbits import export_periodic

            export_periodic(data, folder, depth)

        except Exception:
            traceback.print_exc()

        periodic_checkpoint(
            'Periodic analysis failed · retaining equilibrium results'
        )

    stats['oscillation_seconds'] = (
        time.monotonic()-oscillation_start
    )

    stats['elapsed_seconds'] = time.monotonic()-started

    if dimensions == 2:

        stats.update(
            discovery_seconds=discovery_seconds,
            refinement_seconds=refinement_seconds,
        )

    data['performance'] = stats

    (
        folder/f'performance_H{depth:g}.json'
    ).write_text(
        json.dumps(stats, indent=2),
        encoding='utf-8',
    )

    report(1., f'H={depth:g} m · complete')

    return data


def _atomic_bytes(path, data):
    """Atomically exchange private multiprocessing files."""

    temp = path.with_suffix('.tmp')
    temp.write_bytes(data)

    for attempt in range(8):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.025*(attempt+1))


def _child(p, axes, options, dimensions, folder, work, index):

    last = 0.

    def report(progress, phase):
        nonlocal last

        now = time.monotonic()

        if now-last >= .5 or progress == 1.:

            _atomic_bytes(
                work/f'{index}.progress',
                json.dumps([progress, phase]).encode('utf-8'),
            )

            last = now

    try:

        data = _run_depth(
            p, axes, options, dimensions, folder, report
        )

        _atomic_bytes(
            work/f'{index}.result',
            pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL),
        )

    except BaseException:

        _atomic_bytes(
            work/f'{index}.error',
            traceback.format_exc().encode('utf-8'),
        )

        raise


def _parallel_depths(
    base, depths, axes, options, dimensions, ctx, workers
):

    mpctx = mp.get_context('spawn')

    active = {}
    progress = [0.]*len(depths)
    results = {}
    next_index = 0

    with tempfile.TemporaryDirectory(
        prefix='simp_workers_', dir=ctx.folder
    ) as temp:

        work = Path(temp)

        try:

            while len(results) < len(depths):

                while next_index < len(depths) and len(active) < workers:

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

                            fraction, phase = json.loads(
                                path.read_text(encoding='utf-8')
                            )

                            progress[i] = max(progress[i], fraction)
                            phases.append(phase)

                        except (OSError, ValueError):
                            pass

                    if not process.is_alive():

                        process.join()

                        error = work/f'{i}.error'
                        result = work/f'{i}.result'

                        if process.exitcode != 0 or not result.exists():

                            detail = (
                                error.read_text(encoding='utf-8')
                                if error.exists()
                                else f'exit code {process.exitcode}'
                            )

                            raise RuntimeError(
                                f'Depth H={depths[i]:g} worker failed: '
                                + detail
                            )

                        results[i] = pickle.loads(result.read_bytes())
                        progress[i] = 1.

                        process.close()
                        del active[i]

                ctx.checkpoint(
                    5+80*sum(progress)/len(depths),
                    f'Parallel depths {len(results)}/{len(depths)} · '
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


def run(values, ctx, dimensions):

    ctx.plot_start = 90.

    ctx.checkpoint(0, 'Preparing continuation', force=True)

    base, depths, axes, options = parse_job(values, dimensions)

    started = time.monotonic()

    workers = (
        min(len(depths), int(options.get('depth_workers', 1)))
        if dimensions == 2 else 1
    )

    if workers > 1:

        datasets = _parallel_depths(
            base, depths, axes, options, dimensions, ctx, workers
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

    from scripts.continuation_plots import plot_1d, plot_2d

    ctx.checkpoint(
        90, 'Rendering PNG and PDF figures', force=True
    )

    if dimensions == 1:
        plot_1d(datasets, axes[0], ctx.folder/'figures')
    else:
        plot_2d(datasets, axes, ctx.folder/'figures')

    summary = dict(
        engine='pyoomph native pseudo-arclength',
        dimensions=dimensions,
        depths=depths,
        axes=[vars(a) for a in axes],
        seconds=time.monotonic()-started,
        equation_source='model/shared_eqn.py',
        limitations=(
            'Only numerically verified periodic BVP solutions are shown. '
            'Finite grids, finite initial guesses and numerical '
            'continuation cannot prove global absence of cycles. '
            'Failed searches and incomplete branches remain unresolved. '
            '2D cycles represent verified locations and transects, '
            'not certified regions.'
        ),
        diagnostics=sum(
            len(d['diagnostics']) for d in datasets
        ),
        depth_workers=workers,
        performance={
            str(d['H']): d['performance']
            for d in datasets
        },
    )

    summary['periodic'] = {
        str(d['H']): dict(
            verified_points=len(d.get('oscillations', [])),
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

    (
        ctx.folder/'summary.json'
    ).write_text(
        json.dumps(summary, indent=2),
        encoding='utf-8',
    )

    ctx.checkpoint(
        100, 'Completed native continuation', force=True
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
