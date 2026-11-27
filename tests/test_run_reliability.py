import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core.control import RunContext
from app.core.discovery import discover
from app.core.storage import Repository
from app.core.worker import alive
from model.native_compiler import QuietTCCCompiler


class ReliabilityTests(unittest.TestCase):
    def test_liveness_probe_does_not_kill_process(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            for _ in range(5):
                self.assertTrue(alive(child.pid))
                self.assertIsNone(child.poll())
        finally:
            child.terminate()
            child.wait(timeout=10)
        self.assertFalse(alive(child.pid))

    def test_compiler_hides_subprocess_and_propagates_failure(self):
        result = subprocess.CompletedProcess([], 1, b'compiler error')
        with patch('model.native_compiler.subprocess.run', return_value=result) as run:
            with self.assertRaisesRegex(RuntimeError, 'compiler error'):
                QuietTCCCompiler.call_cmd([sys.executable, '-m', 'tccbox', '-v'])
            if os.name == 'nt':
                self.assertEqual(run.call_args.kwargs['creationflags'], subprocess.CREATE_NO_WINDOW)
                self.assertNotIn('tccbox', run.call_args.args[0])

    def test_progress_sharing_lock_does_not_abort_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = RunContext(tmp)
            with patch.object(Path, 'replace', side_effect=PermissionError), patch('app.core.control.time.sleep'):
                context.checkpoint(20, 'Working', force=True)
            context.checkpoint(100, 'Done', force=True)
            self.assertEqual(json.loads((Path(tmp)/'progress.json').read_text())['progress'], 100)

    def test_delete_result_preserves_other_jobs_and_seed_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = Repository(root/'state/jobs.sqlite', root/'results/runs')
            spec = discover()['timeseries']
            source = repo.create_job('Source', spec, {'duration': 5})
            repo.update_job(source['id'], status='completed')
            repo.register_run_seed(source['id'])
            child = repo.create_job('Child', spec, {'duration': 5}, seed_id='RUN-'+source['id'])
            snapshot = (Path(child['output_dir'])/'config.json').read_bytes()
            repo.delete_result(source['id'])
            self.assertIsNone(repo.get_job(source['id']))
            self.assertIsNone(repo.get_seed('RUN-'+source['id']))
            self.assertFalse(Path(source['output_dir']).exists())
            self.assertEqual((Path(child['output_dir'])/'config.json').read_bytes(), snapshot)
            for status in ('draft', 'queued', 'running', 'paused'):
                repo.update_job(child['id'], status=status)
                with self.assertRaises(ValueError): repo.delete_result(child['id'])
            repo.update_job(child['id'], status='failed')
            with repo.connect() as db:
                db.execute('UPDATE jobs SET output_dir=? WHERE id=?', (str(root), child['id']))
            with self.assertRaises(ValueError): repo.delete_result(child['id'])
            self.assertTrue(root.exists())
