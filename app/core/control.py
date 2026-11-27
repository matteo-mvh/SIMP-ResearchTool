
"""Cooperative pause/cancel control and analysis timing."""

from __future__ import annotations

import json
import sys
import time

from pathlib import Path

from app.core.runtime_estimates import (
    CONTINUATION,
    continuation_snapshot,
    final_profile,
)


class StopRequested(Exception):
    pass


class RunContext:

    def __init__(self, output, analysis_id='', parameters=None):

        self.folder = Path(output)
        self.folder.mkdir(parents=True, exist_ok=True)

        self.analysis_id = analysis_id
        self.parameters = parameters or {}

        self.is_continuation = analysis_id in CONTINUATION

        self.last_write = 0.

        self.stage_seconds = {
            'compute': 0.,
            'plot': 0.,
        }

        if self.is_continuation:
            self.stage_seconds['classification'] = 0.

        self.timing_stage = 'compute'
        self.resume_stage = 'compute'

        self.timing_clock = time.monotonic()

        self.plot_start = 85.

        # Lightweight timing telemetry.
        self.eta_tracker = {}
        self.eta_data = {}
        self.eta_last = float('-inf')
        self.eta_warning_printed = False

    def checkpoint(self, progress, phase='', force=False):
        """Check pause and stop requests before reporting progress."""

        if (self.folder/'stop.request').exists():

            self.progress(
                progress, 'Stopping', force=True
            )

            raise StopRequested()

        paused = False

        while (self.folder/'pause.request').exists():

            if not paused:

                self.progress(
                    progress, 'Paused', force=True
                )

                paused = True

            if (self.folder/'stop.request').exists():
                raise StopRequested()

            time.sleep(.2)

        self.progress(
            progress,
            phase,
            force=force or paused,
        )

    def refresh_eta(self, now, final=False, stage_changed=False):
        """Collect existing scientific diagnostics periodically.

        Timing telemetry must never interrupt the scientific solver.
        """

        if not self.is_continuation:
            return

        if self.timing_stage == 'paused':
            return

        if (
            not final
            and not stage_changed
            and now-self.eta_last < 5.
        ):
            return

        self.eta_last = now

        try:

            self.eta_data = continuation_snapshot(
                self.folder,
                self.parameters,
                self.analysis_id,
                self.eta_tracker,
                self.stage_seconds['compute'],
            )

            # A native attempt may have been running for a long
            # time since the last periodic diagnostic was exported.
            for record in self.eta_data.get('depths', []):

                if record['state'] != 'periodic':
                    continue

                path = (
                    self.folder
                    / f'periodic_H{record["depth"]:g}.json'
                )

                try:
                    age = max(
                        0., time.time()-path.stat().st_mtime
                    )

                    record['periodic']['elapsed'] += age

                except OSError:
                    pass

            if final:

                profile = final_profile(
                    self.eta_data,
                    self.stage_seconds,
                )

                # Learn only from depths whose periodic stage
                # reached its normal computational endpoint.
                healthy = {
                    record['depth']
                    for record in self.eta_data.get('depths', [])
                    if record.get('periodic', {}).get('status') in (
                        'completed',
                        'completed_with_unresolved_searches',
                    )
                }

                profile['depths'] = [
                    record for record in profile['depths']
                    if record['depth'] in healthy
                ]

                if profile['depths']:
                    self.stage_seconds['eta_profile'] = profile

        except Exception as exc:

            if not self.eta_warning_printed:

                print(
                    f'Runtime telemetry unavailable: {exc}',
                    file=sys.stderr,
                )

                self.eta_warning_printed = True

    def progress(self, progress, phase='', force=False):
        """Update timings and publish an atomic progress snapshot."""

        now = time.monotonic()

        previous_stage = self.timing_stage

        # Account for time spent in the previous active stage.
        if previous_stage in self.stage_seconds:

            self.stage_seconds[previous_stage] += (
                now-self.timing_clock
            )

        self.timing_clock = now

        phase_lower = phase.lower()

        native_finished = (
            self.is_continuation
            and phase_lower == 'completed native continuation'
        )

        classification_finished = (
            self.is_continuation
            and phase_lower
            == 'continuation and bifurcation classification completed'
        )

        # Determine the active computational stage.
        if phase == 'Paused':

            if previous_stage != 'paused':
                self.resume_stage = previous_stage

            self.timing_stage = 'paused'

        elif self.is_continuation and (
            native_finished
            or classification_finished
            or phase_lower.startswith('classifying')
        ):

            self.timing_stage = 'classification'

        elif (
            'plotting' in phase_lower
            or 'rendering' in phase_lower
        ):

            if previous_stage != 'plot':
                self.plot_start = progress

            self.timing_stage = 'plot'

        elif progress < self.plot_start:

            self.timing_stage = 'compute'

        elif previous_stage == 'paused':

            self.timing_stage = self.resume_stage

        active_stage = (
            self.resume_stage
            if self.timing_stage == 'paused'
            else self.timing_stage
        )

        stage_changed = (
            previous_stage != self.timing_stage
        )

        # Collect actual per-depth and periodic work counters.
        self.refresh_eta(
            now,
            final=classification_finished,
            stage_changed=stage_changed,
        )

        if not (
            force or classification_finished or stage_changed
        ) and now-self.last_write < .35:

            return

        self.last_write = now

        # Preserve the original runner's numerical progress as input.
        raw_progress = max(0., min(100., float(progress)))

        if active_stage == 'plot':

            stage_progress = (
                (raw_progress-self.plot_start)
                / max(100-self.plot_start, 1.)
            )

        elif active_stage == 'classification':

            # Classification effort is not known in advance.
            stage_progress = (
                1. if classification_finished else 0.
            )

        else:

            stage_progress = (
                raw_progress/max(self.plot_start, 1.)
            )

        displayed_progress = raw_progress

        if self.is_continuation:

            if classification_finished:

                displayed_progress = 100.

            elif active_stage == 'classification':

                # Keep the progress bar below completion until
                # all classification work and figures are finished.
                displayed_progress = 95.

            elif active_stage == 'plot':

                displayed_progress = (
                    self.plot_start
                    + (95-self.plot_start)
                    * max(0., min(1., stage_progress))
                )

        payload = dict(
            progress=displayed_progress,
            phase=phase,
            stage=self.timing_stage,
            active_stage=active_stage,
            stage_progress=max(
                0., min(1., stage_progress)
            ),
            stage_seconds=self.stage_seconds,
            updated=time.time(),
        )

        if self.is_continuation:
            payload['eta'] = self.eta_data

        target = self.folder/'progress.json'
        temporary = self.folder/'progress.tmp'

        temporary.write_text(
            json.dumps(payload),
            encoding='utf-8',
        )

        # Retry brief Windows / OneDrive sharing conflicts.
        for attempt in range(5):

            try:
                temporary.replace(target)
                break

            except PermissionError:

                if attempt == 4:
                    return

                time.sleep(.02*2**attempt)


def load_args():

    import argparse

    parser = argparse.ArgumentParser(
        description='SIMP plugin analysis'
    )

    parser.add_argument(
        '--config', required=True, type=Path
    )

    parser.add_argument(
        '--output', required=True, type=Path
    )

    args = parser.parse_args()

    data = json.loads(
        args.config.read_text(encoding='utf-8')
    )

    parameters = data['parameters']

    context = RunContext(
        args.output,
        analysis_id=data.get('analysis_id', ''),
        parameters=parameters,
    )

    return parameters, context, data
