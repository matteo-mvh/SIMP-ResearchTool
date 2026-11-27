"""Pre-run runtime estimates and GUI integration."""

import json
import math
import sqlite3
import statistics
import time
import tkinter as tk
from pathlib import Path
from weakref import WeakKeyDictionary

from app.core.discovery import validate

from app.core.runtime_estimates import (
    CONTINUATION,
    number,
    duration_range,
    depth_values,
    worker_count,
    engine_signature,
    compatible_history,
    depth_samples,
    remaining_range,
    scheduled_duration,
    workload,
    plotting_work,
    live_label,
)

HISTORY_CACHE = WeakKeyDictionary()
HISTORY_TTL = 8.0


def get_history(repo, analysis):
    """Read timing history without repeated database queries."""
    cache = HISTORY_CACHE.setdefault(repo, {})
    now = time.monotonic()
    if analysis in cache:
        timestamp, rows = cache[analysis]
        if now-timestamp < HISTORY_TTL:
            return rows

    with repo.connect() as db:
        rows = [
            dict(row)
            for row in db.execute(
                """
                SELECT * FROM runtime_history
                WHERE analysis_id=?
                ORDER BY rowid DESC LIMIT 100
                """,
                (analysis,),
            )
        ]

    cache[analysis] = now, rows
    return rows


def decode(value):
    try:
        result = json.loads(value or '{}')
        return result if isinstance(result, dict) else {}

    except (ValueError, TypeError):
        return {}


def uncertainty_range(samples, broad=False):
    samples = [number(value) for value in samples if number(value) > 0]
    if not samples:
        return None

    median = statistics.median(samples)
    if broad or len(samples) == 1:
        return (max(1., min(min(samples), .5*median)), max(max(samples), 2.*median),)

    return (max(1., min(min(samples), .75*median)), max(max(samples), 1.5*median),)


def add_bounds(a, b):
    return (a[0]+b[0], a[1]+b[1],)

# ==============================================================
# Continuation estimates
# ==============================================================

def continuation_forecast(parameters, history, analysis):
    depths = depth_values(parameters)
    workers = worker_count(parameters, analysis)
    signature = engine_signature()
    profiles = compatible_history(parameters, history, signature,)
    broad = False

    # If identical or nearby parameter configurations are unavailable,
    # use other completed runs of the same scientific implementation.
    if not profiles:
        broad = True
        for row in history:
            stages = decode(row.get('stages_json'))
            profile = stages.get('eta_profile', {})
            if profile.get('signature') == signature:
                profiles.append(profile)

            if len(profiles) >= 5:
                break

    if not profiles:
        return None, 'learning'

    estimates = []
    for depth in depths:
        eq = depth_samples(profiles, depth, 'equilibrium_seconds')
        periodic = depth_samples(profiles, depth, 'periodic_seconds')
        depth_broad = broad

        # For a new depth, use available measured depth timings,
        # but increase the uncertainty.
        if not eq:
            depth_broad = True
            eq = [
                record['equilibrium_seconds']
                for profile in profiles
                for record in profile.get('depths', [])
                if number(record.get('equilibrium_seconds')) > 0
            ]

        if not periodic:
            depth_broad = True
            periodic = [
                record['periodic_seconds']
                for profile in profiles
                for record in profile.get('depths', [])
                if number(record.get('periodic_seconds')) > 0
            ]

        eq_bounds = uncertainty_range(eq, depth_broad)
        periodic_bounds = uncertainty_range(periodic, depth_broad)
        if eq_bounds is None or periodic_bounds is None:
            return None, 'learning'

        estimates.append(add_bounds(eq_bounds, periodic_bounds))

    # Independent depth calculations share available worker slots.
    bounds = scheduled_duration([], estimates, workers,)

    # Plotting and classification are sequential post-processing stages.
    overhead = []
    for profile in profiles:
        old_depths = max(1, len(profile.get('depths', [])))
        seconds = (number(profile.get('plot')) + number(profile.get('classification')))
        if seconds > 0:
            overhead.append(seconds*len(depths)/old_depths)

    overhead_bounds = uncertainty_range(overhead, broad)
    if overhead_bounds is not None:
        bounds = add_bounds(bounds, overhead_bounds)

    return bounds, 'rough' if broad else 'learned'

# ==============================================================
# Sensitivity estimates
# ==============================================================

def sensitivity_workers(parameters):
    """The solver submits batches containing up to 16 samples."""
    samples = max(2, int(number(parameters.get('sample_count'), 1000)))
    workers = max(1, int(number(parameters.get('workers'), 1)))
    batches = math.ceil(samples/16)
    return min(workers, batches)


def sensitivity_effort(parameters):
    samples = max(2., number(parameters.get('sample_count'), 1000))
    workers = sensitivity_workers(parameters)

    # Thorough uses 513 root-grid points instead of 257.
    accuracy = (2. if parameters.get('accuracy') == 'Thorough' else 1.)

    # Sublinear worker scaling accounts approximately for overhead.
    return samples*accuracy/workers**.8


def sensitivity_forecast(parameters, history):
    effort = sensitivity_effort(parameters)
    same_accuracy = []
    other_accuracy = []
    plotting = []
    for row in history:
        old = decode(row.get('parameters_json'))
        stages = decode(row.get('stages_json'))
        old_effort = sensitivity_effort(old)
        measured = number(stages.get('compute'))
        if measured <= 0:
            continue

        predicted = measured*effort/old_effort
        target = (
            same_accuracy
            if old.get('accuracy', 'Standard')
            == parameters.get('accuracy', 'Standard')
            else other_accuracy
        )
        target.append(predicted)
        plot = number(stages.get('plot'))
        if plot > 0:
            plotting.append(
                plot*plotting_work(parameters, 'sensitivity')
                / max(plotting_work(old, 'sensitivity'), 1.)
            )

    samples = same_accuracy[:10] or other_accuracy[:10]
    broad = not same_accuracy
    if samples:
        bounds = uncertainty_range(samples, broad)

    else:

        # Initial provisional assumption inherited from the original
        # ETA model: 60 seconds per 1000 single-worker samples.
        # This is not a benchmark measured on the user's computer.
        initial = .06*effort
        bounds = (max(1., .25*initial), max(1., 4.*initial),)
        broad = True

    plot_bounds = uncertainty_range(plotting, broad)
    if plot_bounds is None:
        initial_plot = 3.*plotting_work(parameters, 'sensitivity')
        plot_bounds = (.5*initial_plot, 2.*initial_plot,)

    return (add_bounds(bounds, plot_bounds), 'rough' if broad else 'learned',)

# ==============================================================
# Time series and generic analyses
# ==============================================================

def generic_forecast(parameters, history, analysis):
    samples = []
    for row in history:
        old = decode(row.get('parameters_json'))
        stages = decode(row.get('stages_json'))
        compute = number(stages.get('compute'))
        plot = number(stages.get('plot'))
        if compute > 0:
            seconds = (compute*workload(parameters, analysis) / max(workload(old, analysis), 1e-9))
            seconds += (
                plot*plotting_work(parameters, analysis)
                / max(plotting_work(old, analysis), 1e-9)
            )

        else:
            seconds = (
                number(row.get('seconds'))
                * workload(parameters, analysis)
                / max(workload(old, analysis), 1e-9)
            )

        if seconds > 0:
            samples.append(seconds)

    if samples:
        return uncertainty_range(samples[:10]), 'learned'

    initial = (15.*workload(parameters, analysis) + 3.*plotting_work(parameters, analysis))
    return (max(1., .25*initial), max(1., 4.*initial),), 'rough'


def forecast(parameters, history, analysis):
    if analysis in CONTINUATION:
        return continuation_forecast(parameters, history, analysis)

    if analysis in ('sensitivity', 'sensitivity_demo'):
        return sensitivity_forecast(parameters, history)

    return generic_forecast(parameters, history, analysis)


def estimate_text(parameters, history, analysis):
    bounds, source = forecast(parameters, history, analysis)
    if bounds is None:
        return 'Learning runtime'

    text = '~'+duration_range(bounds)
    if source == 'rough':
        text += ' (rough)'

    return text


def workload_text(parameters, analysis):
    if analysis in ('sensitivity', 'sensitivity_demo'):
        samples = int(number(parameters.get('sample_count'), 1000))
        workers = sensitivity_workers(parameters)
        return f'{samples:,} samples / {workers} workers'

    if analysis in CONTINUATION or analysis == 'timeseries':
        count = len(depth_values(parameters))
        label = f'{count} depth' if count == 1 else f'{count} depths'
        if analysis == 'continuation_2d':
            label += f' / {worker_count(parameters, analysis)} workers'

        return label

    return '—'

# ==============================================================
# Read creation form settings
# ==============================================================

def continuation_values(form):
    values = {key: variable.get() for key, variable in form.controls.items()}
    for prefix in ('x', 'y')[:form.dimensions]:
        parameter = values[prefix+'_parameter']
        values[parameter] = (float(values[prefix+'_min']) + float(values[prefix+'_max']))/2

    values['H'] = float(values['depths'].split(',')[0])
    values = validate(form.spec, values)
    count = int(values['depth_count'])
    depths = [float(value.strip()) for value in values['depths'].split(',')]
    if len(depths) != count or any(not math.isfinite(d) or d <= 0 for d in depths):
        raise ValueError('Incomplete depth configuration')

    return values


def sensitivity_values(form):
    values = {key: variable.get() for key, variable in form.settings.items()}
    ranges = {}
    for key, row in form.rows.items():
        enabled, fixed, low, high = row[:4]
        values[key] = fixed.get()
        if enabled.get():
            lo = float(low.get())
            hi = float(high.get())
            if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi):
                raise ValueError('Invalid sensitivity range')

            ranges[key] = [lo, hi]

    values['ranges'] = json.dumps(ranges)
    return validate(form.spec, values)

# ==============================================================
# Live creation buttons
# ==============================================================

def bind_creation_estimate(parent, repo, spec, read_values, buttons):
    """Refresh button captions as the form settings change.

    Uses only stored runtime history; never launches a solver.
    """
    state = {'after': None, 'label': None,}

    def refresh():
        try:
            values = read_values()
            history = get_history(repo, spec.id)
            text = estimate_text(values, history, spec.id)

        except (ValueError, TypeError, KeyError, OSError, sqlite3.Error, tk.TclError,):
            text = 'Check settings'

        if text != state['label']:
            for widget, original in buttons:
                if widget.winfo_exists():
                    widget.configure(text=f'{original}\nEst. {text}')

            state['label'] = text

        if parent.winfo_exists():
            state['after'] = parent.after(700, refresh)

    def cleanup(event):
        if event.widget is not parent:
            return

        if state['after'] is not None:
            try:
                parent.after_cancel(state['after'])
            except tk.TclError:
                pass

    parent.bind('<Destroy>', cleanup, add='+')
    state['after'] = parent.after_idle(refresh)

# ==============================================================
# Queue table columns
# ==============================================================

def queue_estimate(repo, job):
    parameters = decode(job.get('config_json')).get('parameters', {})
    analysis = job['analysis_id']
    history = get_history(repo, analysis)
    total = estimate_text(parameters, history, analysis)
    if job['status'] in ('running', 'paused'):
        paused = (job['status'] == 'paused' or (Path(job['output_dir'])/'pause.request').exists())
        label = live_label(job, history, paused)

        # Keep the queue cell short. Detailed information remains
        # in the existing footer ETA.
        segments = label.split(' · ')
        eta = next((part for part in segments if part.startswith('ETA')), 'ETA uncertain',)
        if 'uncertain' in eta.lower():
            return f'Left unknown; {total} total'

        return eta.replace('ETA ', '')+' left'

    return total


def queue_columns(repo, job):
    parameters = decode(job.get('config_json')).get('parameters', {})
    try:
        return (workload_text(parameters, job['analysis_id']), queue_estimate(repo, job),)

    except (ValueError, TypeError, KeyError, OSError, sqlite3.Error,):
        return '—', 'Unavailable'
