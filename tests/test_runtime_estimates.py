import json
import stat
import tempfile
import unittest
from pathlib import Path
from app.core.discovery import discover
from app.core.storage import Repository, remove_result_tree
from app.core.runtime_estimates import estimate, estimate_label
from app.core.runtime_estimates import live_label
from app.core.control import RunContext
from unittest.mock import patch


class RuntimeTests(unittest.TestCase):
    def test_stage_eta_available_without_history_and_learns_plotting(self):
        with tempfile.TemporaryDirectory() as tmp:
            job={'analysis_id':'timeseries','config_json':json.dumps({'parameters':{'depth_count':'1','plot_groups':'MS'}}),
                 'output_dir':tmp,'status':'running'}
            self.assertIn('ETA',live_label(job,[]))
            data={'stage':'plot','stage_progress':.5,'stage_seconds':{'compute':90,'plot':10},'updated':100}
            (Path(tmp)/'progress.json').write_text(json.dumps(data))
            history=[{'parameters_json':json.dumps({'depth_count':'1','plot_groups':'MS'}),
                      'seconds':130,'stages_json':json.dumps({'compute':90,'plot':40})}]
            with patch('app.core.runtime_estimates.time.time',return_value=100):
                label=live_label(job,history)
            self.assertIn('20 s',label)  # Plot total blends historical 40 s and observed 20 s.
            self.assertIn('learned stage timings',label)
            self.assertIn('after resume',live_label(job,history,True))

    def test_stage_clock_excludes_pause(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('app.core.control.time.monotonic',return_value=0):ctx=RunContext(tmp)
            for clock,progress,phase in ((10,50,'Integrating'),(12,50,'Paused'),(100,50,'Integrating'),
                                         (110,85,'Plotting MS'),(115,100,'Completed')):
                with patch('app.core.control.time.monotonic',return_value=clock):
                    ctx.progress(progress,phase,force=True)
            self.assertEqual(ctx.stage_seconds,{'compute':22.,'plot':5.})
    def test_remove_readonly_directory_and_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'result'
            folder.mkdir()
            file=folder/'plot.png'
            file.write_bytes(b'test')
            file.chmod(stat.S_IREAD)
            folder.chmod(stat.S_IREAD)
            remove_result_tree(folder)
            self.assertFalse(folder.exists())

    def test_learning_survives_result_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            repo=Repository(root/'jobs.sqlite',root/'runs')
            job=repo.create_job('History',discover()['timeseries'],{'duration':5})
            self.assertIsNone(repo.estimate_runtime(job))
            repo.update_job(job['id'],status='completed',started_at='2026-01-01T00:00:00+00:00',
                            ended_at='2026-01-01T00:02:00+00:00')
            repo.learn_completed_runtimes()
            repo.learn_completed_runtimes()
            self.assertEqual(repo.estimate_runtime(job),(120,1))
            repo.delete_result(job['id'])
            self.assertEqual(repo.estimate_runtime(job),(120,1))

    def test_depth_scaling_and_unknown_history(self):
        parameters={'depth_count':'1','discovery_points':9,'max_steps':600,'x_resolution':101}
        history=[{'parameters_json':json.dumps(parameters),'seconds':100}]
        parameters['depth_count']='4'
        job={'analysis_id':'continuation_1d','config_json':json.dumps({'parameters':parameters}), 'progress':50}
        self.assertEqual(estimate(job,history),(400,1))
        self.assertIn('3 min remaining',estimate_label(job,estimate(job,history)))
        self.assertIn('Paused',estimate_label(job,estimate(job,history),True))
        self.assertIsNone(estimate(job,[]))
