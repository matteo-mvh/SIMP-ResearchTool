
"""Runtime estimation using measured work rather than GUI progress."""

import hashlib
import json
import math
import statistics
import time

from datetime import datetime
from functools import lru_cache
from pathlib import Path


CONTINUATION = ('continuation_1d', 'continuation_2d')

ROOT = Path(__file__).resolve().parents[2]

ETA_VERSION = 1


# ==============================================================
# General helpers
# ==============================================================

def number(value, default=0.):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (ValueError, TypeError):
        return default


def parameters_of(job):
    try:
        return json.loads(job['config_json']).get('parameters', {})
    except (KeyError, ValueError, TypeError):
        return {}


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def measured_seconds(job):
    try:
        seconds = (
            datetime.fromisoformat(job['ended_at'])
            - datetime.fromisoformat(job['started_at'])
        ).total_seconds()

        return seconds if math.isfinite(seconds) and seconds > 0 else None

    except (ValueError, TypeError, KeyError):
        return None


def duration(seconds):
    seconds = max(0., seconds)

    if seconds < 60:
        return f'{max(1, round(seconds))} s'

    if seconds < 3600:
        return f'{round(seconds/60)} min'

    return f'{seconds/3600:.1f} h'


def duration_range(bounds):
    return f'{duration(bounds[0])}–{duration(bounds[1])}'


def depth_values(parameters):
    values = parameters.get('depths', '4')

    if isinstance(values, str):
        values = values.split(',')

    depths = [
        number(value, -1.)
        for value in values
    ]

    return [depth for depth in depths if depth > 0] or [4.]


def worker_count(parameters, analysis):

    if analysis == 'continuation_1d':
        return 1

    return max(1, min(
        len(depth_values(parameters)),
        int(number(parameters.get('depth_workers', 1), 1)),
    ))


@lru_cache(maxsize=1)
def engine_signature():
    """Invalidate learned runtimes after scientific code changes."""

    paths = [
        *ROOT.glob('model/*.py'),
        *ROOT.glob('model/equilibria/**/*.py'),
        *ROOT.glob('scripts/continuation_*.py'),
        ROOT/'scripts/bifurcation_classification.py',
    ]

    digest = hashlib.sha256()

    for path in sorted(set(paths)):

        if not path.is_file():
            continue

        try:
            digest.update(
                str(path.relative_to(ROOT)).encode('utf-8')
            )
            digest.update(path.read_bytes())
        except OSError:
            continue

    return f'{ETA_VERSION}:{digest.hexdigest()[:20]}'


# ==============================================================
# Existing generic runtime-estimation interfaces
# ==============================================================

def workload(parameters, analysis):

    def n(key, default):
        return max(1., number(parameters.get(key), default))

    if analysis in CONTINUATION:

        dimensions = 2 if analysis.endswith('2d') else 1

        discovery = n('discovery_points', 9)/9

        if dimensions == 2:
            discovery *= n('transects', 5)/5

        tracing = n('x_resolution', 101)/101

        if dimensions == 2:
            tracing += n('y_resolution', 101)/101
            tracing /= 2

        refinement = 1.

        if dimensions == 2:
            refinement += (
                .2*max(0., number(parameters.get('refinement_rounds')))
                * n('refinement_searches', 12)/12
            )

        return (
            n('depth_count', 1)
            / worker_count(parameters, analysis)
            * (.5*discovery+.5*tracing)
            * refinement
        )

    if analysis == 'timeseries':
        return n('depth_count', 1)*(
            n('duration', 1000)/1000
            + n('samples', 1001)/1001
        )

    if analysis == 'sensitivity':
        return (
            n('sample_count', 1000)/1000
            / n('workers', 1)
            * (2 if parameters.get('accuracy') == 'Thorough' else 1)
        )

    for key in ('sample_count', 'points', 'days'):
        if key in parameters:
            return n(key, 1)

    return 1.


def parameter_distance(current, previous):
    """Similarity metric for ecological and numerical parameters."""

    ignored = {
        'seed_id', 'depths', 'depth_count', 'depth_workers',
        'workers',
    }

    distance = 0.

    for key in current.keys() | previous.keys():

        if key in ignored:
            continue

        a = current.get(key)
        b = previous.get(key)

        if a == b:
            continue

        try:
            a, b = float(a), float(b)

            if not math.isfinite(a) or not math.isfinite(b):
                distance += 1.
                continue

            distance += abs(a-b)/max(abs(a), abs(b), 1e-9)

        except (ValueError, TypeError):
            distance += 1.

    return distance


def estimate(job, history):
    """Legacy historical estimate; retained for existing callers."""

    parameters = parameters_of(job)
    analysis = job['analysis_id']

    candidates = []

    for row in history:

        try:
            old = json.loads(row['parameters_json'])
        except (KeyError, ValueError, TypeError):
            continue

        seconds = (
            number(row.get('seconds'))
            * workload(parameters, analysis)
            / max(workload(old, analysis), 1e-9)
        )

        if seconds > 0:
            candidates.append((
                parameter_distance(parameters, old),
                seconds,
            ))

    if not candidates:
        return None

    candidates.sort()
    nearest = candidates[:5]

    return statistics.median(
        seconds for _, seconds in nearest
    ), len(nearest)


def estimate_label(job, prediction, paused=False):

    if paused:
        return 'Paused · time estimate resumes with the run'

    if prediction is None:
        return 'Time estimate: learning from completed runs'

    total, count = prediction

    fraction = min(.99, max(
        0., number(job.get('progress'))/100
    ))

    return (
        f'About {duration(total*(1-fraction))} remaining'
        f' · {duration(total)} total'
        f' · based on {count} similar run(s)'
    )


def plotting_work(parameters, analysis):

    if analysis == 'sensitivity':

        try:
            ranges = json.loads(parameters.get('ranges', '{}'))
        except (ValueError, TypeError):
            ranges = {}

        return 5+2*len(ranges)

    depths = max(1, int(number(
        parameters.get('depth_count'), 1
    )))

    if analysis == 'timeseries':
        figures = len(str(
            parameters.get('plot_groups', 'MS, P, N')
        ).split(','))
    else:
        figures = depths+1 if analysis == 'continuation_1d' else 3

    return figures*(1+.25*(depths-1))


# ==============================================================
# Read actual continuation work
# ==============================================================

def cached_json(path, cache):
    """Avoid repeatedly reading unchanged scientific output files."""

    path = Path(path)
    key = str(path)

    try:
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return {}

    previous = cache.get(key)

    if previous and previous[0] == stamp:
        return previous[1]

    data = read_json(path)

    if data:
        cache[key] = stamp, data
        return data

    # A file may be between writes. Retain the last valid snapshot.
    return previous[1] if previous else {}


def uv_key(values):
    return tuple(round(float(v), 11) for v in values)


def periodic_metrics(payload):
    """Extract useful work counters from periodic diagnostics."""

    coverage = payload.get('coverage', {})
    locations = payload.get('locations', [])
    attempts = payload.get('attempts', [])

    incomplete = {
        uv_key(uv)
        for uv in coverage.get('incomplete_locations', [])
    }

    completed = {
        uv_key(location['uv'])
        for location in locations
        if 'uv' in location
        and uv_key(location['uv']) not in incomplete
    }

    scheduled = int(number(
        coverage.get('scheduled_locations'), len(locations)
    ))

    done = int(number(
        coverage.get('completed_locations'), len(completed)
    ))

    discovery_cost = 0.
    costly_locations = set()
    trace_seconds = 0.

    for attempt in attempts:

        seconds = max(0., number(attempt.get('seconds')))

        if attempt.get('kind') == 'trace':
            trace_seconds += seconds
            continue

        if attempt.get('kind') not in ('manual', 'reuse'):
            continue

        uv = attempt.get('uv')

        if uv is None:
            continue

        key = uv_key(uv)

        if key in completed:
            discovery_cost += seconds

            if seconds > 0:
                costly_locations.add(key)

    queue_remaining = max(0, scheduled-done)

    # Do not extrapolate from just one or two expensive locations.
    seconds_per_location = None

    if done >= 5 and len(costly_locations) >= 3:
        seconds_per_location = discovery_cost/done

    # This estimates ONLY the currently known discovery queue.
    # New branches and refinement locations can add further work.
    queue_seconds = (
        queue_remaining*seconds_per_location
        if seconds_per_location is not None
        else None
    )

    return dict(
        status=payload.get('status', 'running'),
        scheduled=scheduled,
        completed=done,
        checked=int(number(coverage.get('checked_locations'))),
        attempts=len(attempts),
        branches=len(payload.get('branches', [])),
        pending_branches=len(coverage.get('pending_branches', [])),
        elapsed=max(0., number(coverage.get('elapsed_seconds'))),
        discovery_seconds=discovery_cost,
        trace_seconds=trace_seconds,
        queue_seconds=queue_seconds,
    )


def continuation_snapshot(
    folder, parameters, analysis, tracker, compute_elapsed
):
    """Summarize per-depth work for progress.json.

    Called by the analysis process, not by the Tkinter GUI.
    """

    folder = Path(folder)

    cache = tracker.setdefault('cache', {})
    starts = tracker.setdefault('starts', {})

    records = []

    for depth in depth_values(parameters):

        suffix = f'H{depth:g}'

        performance = cached_json(
            folder/f'performance_{suffix}.json', cache
        )

        periodic = cached_json(
            folder/f'periodic_{suffix}.json', cache
        )

        record = dict(
            depth=depth,
            state='waiting',
        )

        if periodic:

            metrics = periodic_metrics(periodic)

            record['periodic'] = metrics

            if metrics['status'] == 'running':
                record['state'] = 'periodic'

        if performance and 'oscillation_seconds' in performance:

            record.update(
                state='done',
                equilibrium_seconds=number(
                    performance.get('equilibrium_seconds')
                ),
                periodic_seconds=number(
                    performance.get('oscillation_seconds')
                ),
            )

        records.append(record)

    # The runner starts depth workers in configuration order.
    unfinished = [
        record for record in records
        if record['state'] != 'done'
    ]

    workers = worker_count(parameters, analysis)

    active = unfinished[:workers]

    for record in active:

        key = str(record['depth'])
        starts.setdefault(key, compute_elapsed)

        if record['state'] == 'waiting':
            record['state'] = 'equilibrium'

        record['active_elapsed'] = max(
            0., compute_elapsed-starts[key]
        )

    return dict(
        signature=tracker.setdefault(
            'signature', engine_signature()
        ),
        workers=workers,
        depths=records,
    )


def final_profile(snapshot, stage_seconds):
    """Compact timing record persisted in the existing history table."""

    return dict(
        signature=snapshot.get('signature'),
        depths=[
            {
                'depth': r['depth'],
                'equilibrium_seconds': r['equilibrium_seconds'],
                'periodic_seconds': r['periodic_seconds'],
            }
            for r in snapshot.get('depths', [])
            if r['state'] == 'done'
            and r.get('periodic', {}).get('status') not in (
                'failed', 'disabled', 'interrupted',
            )
        ],
        plot=number(stage_seconds.get('plot')),
        classification=number(stage_seconds.get('classification')),
    )


# ==============================================================
# Version-compatible historical timing
# ==============================================================

def compatible_history(parameters, history, signature):
    """Select nearby runs using the same scientific implementation."""

    candidates = []

    for row in history:

        try:
            old = json.loads(row['parameters_json'])
            stages = json.loads(row.get('stages_json') or '{}')
        except (KeyError, ValueError, TypeError):
            continue

        profile = stages.get('eta_profile', {})

        if profile.get('signature') != signature:
            continue

        if any(
            parameters.get(key) != old.get(key)
            for key in ('x_parameter', 'y_parameter')
        ):
            continue

        distance = parameter_distance(parameters, old)

        # Very different parameter combinations are not reliable
        # predictors of bifurcation and orbit complexity.
        if distance > 3.:
            continue

        candidates.append((distance, profile))

    candidates.sort(key=lambda item: item[0])

    return [profile for _, profile in candidates[:5]]


def depth_samples(profiles, depth, key):
    """Prefer historical calculations at approximately the same depth."""

    matches = []

    for profile in profiles:

        for record in profile.get('depths', []):

            old_depth = number(record.get('depth'))

            if old_depth <= 0:
                continue

            relative = abs(depth-old_depth)/max(depth, old_depth)

            if relative > .30:
                continue

            seconds = number(record.get(key))

            if seconds > 0:
                matches.append((relative, seconds))

    matches.sort()

    return [seconds for _, seconds in matches[:8]]


def remaining_range(samples, elapsed=0.):
    """Heuristic range, conditional on the stage still being active.

    Never report a near-zero ETA simply because the historical
    predicted duration has already been exceeded.
    """

    remaining = [
        number(seconds)-elapsed
        for seconds in samples
        if number(seconds) > elapsed
    ]

    if not remaining:
        return None

    median = statistics.median(remaining)

    return (
        max(1., min(min(remaining), .75*median)),
        max(max(remaining), 1.5*median),
    )


def add_ranges(*ranges):

    if any(bounds is None for bounds in ranges):
        return None

    return (
        sum(bounds[0] for bounds in ranges),
        sum(bounds[1] for bounds in ranges),
    )


def scheduled_duration(active, waiting, workers):
    """Estimate wall time, accounting for parallel depth workers."""

    if any(bounds is None for bounds in active+waiting):
        return None

    output = []

    for index in (0, 1):

        lanes = [0.]*workers

        for lane, bounds in enumerate(active):
            lanes[lane] = bounds[index]

        for bounds in waiting:

            lane = min(
                range(workers), key=lambda k: lanes[k]
            )

            lanes[lane] += bounds[index]

        output.append(max(lanes))

    return tuple(output)


def continuation_remaining(snapshot, profiles, stage, elapsed):
    """Estimate remaining wall time from depth states and phase timings."""

    plot_samples = [
        profile['plot']
        for profile in profiles
        if number(profile.get('plot')) > 0
    ]

    classification_samples = [
        profile['classification']
        for profile in profiles
        if number(profile.get('classification')) > 0
    ]

    if stage == 'classification':
        return remaining_range(
            classification_samples, elapsed
        )

    if stage == 'plot':
        return add_ranges(
            remaining_range(plot_samples, elapsed),
            remaining_range(classification_samples),
        )

    active = []
    waiting = []

    for record in snapshot.get('depths', []):

        state = record['state']
        depth = record['depth']

        if state == 'done':
            continue

        eq = depth_samples(
            profiles, depth, 'equilibrium_seconds'
        )

        periodic = depth_samples(
            profiles, depth, 'periodic_seconds'
        )

        if state == 'waiting':

            waiting.append(add_ranges(
                remaining_range(eq),
                remaining_range(periodic),
            ))

        elif state == 'equilibrium':

            active.append(add_ranges(
                remaining_range(
                    eq, record.get('active_elapsed', 0.)
                ),
                remaining_range(periodic),
            ))

        elif state == 'periodic':

            metrics = record.get('periodic', {})

            # The remaining length of a newly discovered orbit
            # branch cannot be inferred from location progress.
            if metrics.get('pending_branches', 0):
                active.append(None)
                continue

            bounds = remaining_range(
                periodic, metrics.get('elapsed', 0.)
            )

            known_queue = metrics.get('queue_seconds')

            if bounds and known_queue is not None:
                bounds = (
                    max(bounds[0], .5*known_queue),
                    max(bounds[1], 2*known_queue),
                )

            active.append(bounds)

    compute = scheduled_duration(
        active, waiting, snapshot.get('workers', 1)
    )

    return add_ranges(
        compute,
        remaining_range(plot_samples),
        remaining_range(classification_samples),
    )


# ==============================================================
# Continuation-specific GUI label
# ==============================================================

def continuation_label(job, history, data, paused):

    parameters = parameters_of(job)
    analysis = job['analysis_id']

    snapshot = data.get('eta', {})

    profiles = compatible_history(
        parameters,
        history,
        snapshot.get('signature') or engine_signature(),
    )

    if job.get('status') in ('queued', 'draft'):

        snapshot = dict(
            workers=worker_count(parameters, analysis),
            depths=[
                dict(depth=depth, state='waiting')
                for depth in depth_values(parameters)
            ],
        )

    stage = data.get(
        'active_stage', data.get('stage', 'compute')
    )

    phase = str(data.get('phase', job.get('phase', '')))

    if 'classifying' in phase.lower():
        stage = 'classification'

    elapsed = number(
        data.get('stage_seconds', {}).get(stage)
    )

    if (
        data and not paused
        and job.get('status') == 'running'
    ):
        elapsed += max(
            0., time.time()-number(data.get('updated'), time.time())
        )

    bounds = (
        continuation_remaining(
            snapshot, profiles, stage, elapsed
        )
        if snapshot.get('depths')
        else None
    )

    details = []

    if stage == 'classification':
        details.append('bifurcation classification')

    elif stage == 'plot':
        details.append('rendering figures')

    else:

        periodic = [
            record['periodic']
            for record in snapshot.get('depths', [])
            if record['state'] == 'periodic'
            and 'periodic' in record
        ]

        if periodic:

            done = sum(p['completed'] for p in periodic)
            total = sum(p['scheduled'] for p in periodic)
            attempts = sum(p['attempts'] for p in periodic)
            pending = sum(p['pending_branches'] for p in periodic)

            details.append(f'periodic: {done}/{total} locations')
            details.append(f'{attempts} native attempts')

            if pending:
                details.append(f'{pending} branches pending')

            elif bounds is None:

                queue = [
                    p['queue_seconds'] for p in periodic
                    if p['queue_seconds'] is not None
                ]

                if queue:

                    # These depths run independently, so the longest
                    # known queue determines their current wall time.
                    remaining = max(queue)

                    details.append(
                        'known discovery queue ~'
                        + duration_range((
                            .5*remaining, 2*remaining
                        ))
                    )

        else:
            details.append('equilibrium continuation')

    if bounds is not None:
        label = (
            f'ETA ~ {duration_range(bounds)}'
            f' · {len(profiles)} comparable runs'
        )

    else:
        label = 'ETA uncertain'

    if paused:
        label = 'Paused · '+label+' after resume'

    if details:
        label += ' · '+' · '.join(details)

    return label


# ==============================================================
# Existing non-continuation ETA
# ==============================================================

def generic_live_label(job, history, data, paused):

    parameters = parameters_of(job)
    analysis = job['analysis_id']

    totals = {
        'compute': (
            15 if analysis == 'timeseries' else 60
        )*workload(parameters, analysis),

        'plot': 3*plotting_work(parameters, analysis),
    }

    learned = 0

    for stage in totals:

        samples = []

        for row in history:

            try:
                stages = json.loads(
                    row.get('stages_json') or '{}'
                )
                old = json.loads(row['parameters_json'])
            except (KeyError, ValueError, TypeError):
                continue

            seconds = number(stages.get(stage))

            if seconds <= 0:
                continue

            scale = plotting_work if stage == 'plot' else workload

            samples.append(
                seconds*scale(parameters, analysis)
                / max(scale(old, analysis), 1e-9)
            )

        if samples:
            totals[stage] = statistics.median(samples[:10])
            learned = max(learned, len(samples[:10]))

    if not learned:

        previous = estimate(job, history)

        if previous:
            totals['compute'] = max(
                1., previous[0]-totals['plot']
            )

    stage = data.get(
        'active_stage', data.get('stage', 'compute')
    )

    if stage not in totals:
        stage = 'plot' if number(data.get('progress')) >= 85 else 'compute'

    fraction = min(1., max(
        0., number(data.get('stage_progress'))
    ))

    elapsed = number(
        data.get('stage_seconds', {}).get(stage)
    )

    if data and not paused and job.get('status') == 'running':
        elapsed += max(
            0., time.time()-number(data.get('updated'), time.time())
        )

    predicted = totals[stage]

    if fraction > .01:
        weight = min(.85, fraction)
        predicted = (
            (1-weight)*predicted
            + weight*elapsed/fraction
        )

    remaining = max(
        1., predicted-elapsed,
        predicted*(1-fraction)*.1,
    )

    if stage == 'compute':
        remaining += totals['plot']

    if job.get('status') in ('draft', 'queued'):
        remaining = sum(totals.values())

    source = (
        f'learned stage timings ({learned} runs)'
        if learned else 'provisional; refining live'
    )

    prefix = 'Paused · ' if paused else ''
    suffix = ' after resume' if paused else ''

    return (
        f'{prefix}ETA ≈ {duration(remaining)}{suffix}'
        f' · {source}'
    )


def live_label(job, history, paused=False):
    """Public interface used by Repository.runtime_label()."""

    output = job.get('output_dir')

    data = read_json(
        Path(output)/'progress.json'
    ) if output else {}

    if job['analysis_id'] in CONTINUATION:

        return continuation_label(
            job, history, data, paused
        )

    return generic_live_label(
        job, history, data, paused
    )
