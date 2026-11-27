
"""Regression tests for continuation runtime estimation."""

import json
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from app.core.control import RunContext

from app.core.runtime_estimates import (
    periodic_metrics,
    compatible_history,
    remaining_range,
    scheduled_duration,
    continuation_remaining,
    live_label,
)


class ContinuationETATests(unittest.TestCase):

    def test_periodic_discovery_cost(self):
        """Trace time must not be confused with grid discovery time."""

        locations = [
            dict(uv=[i/10])
            for i in range(5)
        ]

        payload = dict(
            status='running',
            locations=locations,
            branches=[],
            coverage=dict(
                scheduled_locations=20,
                checked_locations=5,
                completed_locations=5,
                incomplete_locations=[],
                pending_branches=[],
            ),
            attempts=[
                dict(kind='manual', uv=[0.], seconds=10.),
                dict(kind='manual', uv=[.1], seconds=10.),
                dict(kind='manual', uv=[.2], seconds=10.),
                dict(kind='trace', uv=[.1], seconds=100.),
            ],
        )

        result = periodic_metrics(payload)

        self.assertEqual(result['discovery_seconds'], 30.)
        self.assertEqual(result['trace_seconds'], 100.)

        # 30 seconds / 5 completed locations = 6 seconds/location.
        # 15 remaining locations * 6 seconds = 90 seconds.
        self.assertEqual(result['queue_seconds'], 90.)

    def test_parallel_depth_workers(self):

        active = [(10., 20.), (40., 60.)]
        waiting = [(20., 30.)]

        result = scheduled_duration(
            active, waiting, workers=2
        )

        self.assertEqual(result, (40., 60.))

    def test_serial_depth_workers(self):

        result = scheduled_duration(
            [(10., 20.)],
            [(20., 30.)],
            workers=1,
        )

        self.assertEqual(result, (30., 50.))

    def test_exceeded_prediction_remains_uncertain(self):
        """Never report 1 second remaining after exceeding all history."""

        self.assertIsNone(
            remaining_range(
                [10., 20.],
                elapsed=30.,
            )
        )

    def test_old_solver_history_is_rejected(self):

        parameters = dict(
            x_parameter='N_wc',
            x_min=1.,
            x_max=100.,
        )

        history = [
            dict(
                parameters_json=json.dumps(parameters),
                stages_json=json.dumps({
                    'eta_profile': {
                        'signature': 'old-solver',
                    }
                }),
            ),
            dict(
                parameters_json=json.dumps(parameters),
                stages_json=json.dumps({
                    'eta_profile': {
                        'signature': 'current-solver',
                    }
                }),
            ),
        ]

        profiles = compatible_history(
            parameters,
            history,
            'current-solver',
        )

        self.assertEqual(len(profiles), 1)

        self.assertEqual(
            profiles[0]['signature'],
            'current-solver',
        )

    def test_pending_branch_has_no_false_eta(self):

        snapshot = dict(
            workers=1,
            depths=[
                dict(
                    depth=4.,
                    state='periodic',
                    periodic=dict(
                        pending_branches=1,
                        elapsed=10.,
                    ),
                )
            ],
        )

        profiles = [
            dict(
                depths=[
                    dict(
                        depth=4.,
                        equilibrium_seconds=20.,
                        periodic_seconds=60.,
                    )
                ],
                plot=5.,
                classification=10.,
            )
        ]

        result = continuation_remaining(
            snapshot,
            profiles,
            stage='compute',
            elapsed=30.,
        )

        self.assertIsNone(result)

    def test_no_history_reports_uncertainty(self):

        with tempfile.TemporaryDirectory() as folder:

            job = dict(
                analysis_id='continuation_2d',
                status='running',
                output_dir=folder,
                config_json=json.dumps({
                    'parameters': {
                        'depths': '4',
                    }
                }),
            )

            label = live_label(job, [])

            self.assertIn('ETA uncertain', label)

    def test_classification_is_separate_stage(self):

        with tempfile.TemporaryDirectory() as folder:

            with patch(
                'app.core.control.time.monotonic',
                return_value=0.,
            ):

                ctx = RunContext(
                    folder,
                    analysis_id='continuation_1d',
                    parameters={'depths': '4'},
                )

            ctx.plot_start = 90.

            stages = [
                (10., 90., 'Rendering PNG and PDF figures'),
                (20., 100., 'Completed native continuation'),
                (25., 100., 'Classifying bifurcations · H=4 m'),
            ]

            for clock, progress, phase in stages:

                with patch(
                    'app.core.control.time.monotonic',
                    return_value=clock,
                ):

                    ctx.progress(
                        progress, phase, force=True
                    )

            payload = json.loads(
                (Path(folder)/'progress.json').read_text()
            )

            self.assertEqual(
                payload['stage'], 'classification'
            )

            self.assertLess(
                payload['progress'], 100.
            )

            with patch(
                'app.core.control.time.monotonic',
                return_value=30.,
            ):

                ctx.progress(
                    100.,
                    'Continuation and bifurcation classification completed',
                    force=True,
                )

            payload = json.loads(
                (Path(folder)/'progress.json').read_text()
            )

            self.assertEqual(payload['progress'], 100.)

            self.assertEqual(
                ctx.stage_seconds['compute'], 10.
            )

            self.assertEqual(
                ctx.stage_seconds['plot'], 10.
            )

            self.assertEqual(
                ctx.stage_seconds['classification'], 10.
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)
