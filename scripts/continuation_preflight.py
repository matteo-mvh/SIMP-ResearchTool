"""Save equilibrium results and classify bifurcations before orbit tracking."""

import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.bifurcation_classification import (
    classify_event,
    audit_1d,
    write_csv,
    plot_classification,
)


def prepare_equilibrium_results(data, base, axes, folder):
    folder = Path(folder)
    dimensions = len(axes)
    suffix = f'H{base.H:g}'
    native_events = [
        classify_event(
            dict(row, source='native_event'),
            base,
            axes,
        )
        for row in data.get('events', [])
    ]
    curves = []
    for curve in data.get('curves', []):
        curves.append([
            classify_event(
                dict(row, source='native_curve'),
                base,
                axes,
            )
            for row in curve
        ])

    branch_rows = [
        dict(row, branch=i, point=j)
        for i, path in enumerate(data.get('paths', []), 1)
        for j, row in enumerate(path)
    ]
    added = []
    failures = []
    if dimensions == 1:
        added, failures = audit_1d(branch_rows, native_events, base, axes[0],)

    classified_events = native_events+added

    # Preserve existing native events, but introduce independently
    # discovered events only after successful classification.
    data['events'] = native_events+[
        row for row in added
        if row['classification_status'] == 'verified'
    ]
    if dimensions == 2:
        data['curves'] = curves

    curve_rows = [
        dict(row, curve=i, point=j)
        for i, curve in enumerate(curves, 1)
        for j, row in enumerate(curve)
    ]
    write_csv(folder/f'events_{suffix}.csv', data['events'],)
    write_csv(folder/f'bifurcations_{suffix}.csv', classified_events,)
    write_csv(folder/f'bifurcation_curves_{suffix}.csv', curve_rows,)
    write_csv(folder/f'bifurcation_audit_failures_{suffix}.csv', failures,)
    figures = folder/'figures'
    figures.mkdir(exist_ok=True)

    # These figures remain available even if the subsequent periodic
    # calculation is cancelled or encounters a native failure.
    plot_classification(classified_events, curve_rows, branch_rows, axes, base.H, figures,)
    from scripts.continuation_plots import plot_1d, plot_2d
    with TemporaryDirectory(prefix='equilibrium_preview_', dir=folder) as directory:
        preview = Path(directory)
        if dimensions == 1:
            plot_1d([data], axes[0], preview)
        else:
            plot_2d([data], axes, preview)

        for path in preview.iterdir():
            target = figures/f'preview_{suffix}__{path.name}'
            path.replace(target)

    summary = dict(
        native_event_candidates=len(native_events),
        independent_candidates=len(added),
        classified_curve_points=len(curve_rows),
        event_classifications=dict(Counter(
            row['classification'] for row in classified_events
        )),
        curve_classifications=dict(Counter(
            row['classification'] for row in curve_rows
        )),
        audit_correction_failures=len(failures),
        verified_events=sum(
            row['classification_status'] == 'verified'
            for row in classified_events
        ),
    )
    (folder/f'bifurcation_summary_{suffix}.json').write_text(
        json.dumps(summary, indent=2),
        encoding='utf-8',
    )
    return summary
