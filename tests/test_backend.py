"""Run with: python -m unittest discover -s tests -v"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from app.core.discovery import discover, validate
from app.core.storage import Repository
from app.core.worker import request, alive
from app.core.paths import ROOT

class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo=Repository(self.root/'state'/'jobs.sqlite',self.root/'results'/'runs')
        self.analyses=discover(ROOT/'tests/fixtures/scripts')

    def tearDown(self): self.temp.cleanup()

    def test_discovers_manifests_without_importing_analysis(self):
        self.assertTrue({'sensitivity_demo','continuation_1d_demo','continuation_2d_demo'}.issubset(self.analyses))
        self.assertEqual(self.analyses['sensitivity_demo'].parameters['sample_count']['default'],80)

    def test_parameter_validation(self):
        a=self.analyses['sensitivity_demo']
        self.assertEqual(validate(a,{'sample_count':'12'})['sample_count'],12)
        with self.assertRaises(ValueError): validate(a,{'sample_count':'-3'})
        with self.assertRaises(ValueError): validate(a,{'invalid':'10'})

    def test_create_is_draft_and_queue_is_explicit(self):
        j=self.repo.create_job('A test',self.analyses['continuation_1d_demo'],{'points':5,'delay_ms':0})
        self.assertEqual(j['status'],'draft')
        self.assertTrue((Path(j['output_dir'])/'config.json').exists())
        self.repo.enqueue([j['id']])
        self.assertEqual(self.repo.get_job(j['id'])['status'],'queued')

    def test_full_worker_sequence_and_seed_registry(self):
        # worker is an independent process, and tests use isolated data directory
        a=self.analyses['sensitivity_demo']
        j=self.repo.create_job('Sweep',a,{'sample_count':6,'delay_ms':0,'random_seed':22})
        self.repo.enqueue([j['id']])
        env=os.environ.copy()
        env['SIMP_DATA_DIR']=str(self.root)
        env['SIMP_SCRIPTS_DIR']=str(ROOT/'tests/fixtures/scripts')
        # Queue job in isolated worker database, pointing to same directory.
        target=Repository(self.root/'state'/'jobs.sqlite',self.root/'results'/'runs')
        p=subprocess.Popen([sys.executable,str(ROOT/'start_worker.py'),'--once'],cwd=ROOT,env=env,
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        deadline=time.monotonic()+45
        try:
            while p.poll() is None and time.monotonic()<deadline:
                alive(p.pid)  # Same repeated liveness probe used by the GUI.
                time.sleep(.05)
            _,stderr=p.communicate(timeout=2)
        finally:
            if p.poll() is None:
                p.kill()
                p.wait(timeout=5)
        self.assertEqual(p.returncode,0,stderr)
        done=target.get_job(j['id'])
        self.assertEqual(done['status'],'completed')
        self.assertEqual(len(target.list_seeds()),7)  # six samples plus one completed-run snapshot
        self.assertIsNotNone(target.get_seed('RUN-'+j['id']))
        self.assertTrue((Path(done['output_dir'])/'figures'/'sensitivity_scatter.png').exists())


    def test_paginated_large_seed_library_and_snapshot(self):
        """Several thousand samples are indexed, but only small pages enter the GUI."""
        job=self.repo.create_job('Huge sweep',self.analyses['sensitivity_demo'],{'sample_count':6000})
        csvfile=self.root/'many_seeds.csv'
        import csv
        with csvfile.open('w',newline='',encoding='utf-8') as fp:
            writer=csv.writer(fp)
            writer.writerow(['seed_id','kw','g_M','metric__bistability','metric__delta_S'])
            for i in range(6000):
                writer.writerow([f'S-abc12345-{i+1:06d}',0.2+i/100000,0.01, i%2, float(i)])
        self.assertEqual(self.repo.register_seeds(job['id'],csvfile),6000)
        self.assertEqual(self.repo.seed_count(),6000)
        self.assertEqual(self.repo.list_seed_sources()[0]['seed_count'],6000)
        n,rows=self.repo.find_seeds(source_job_id=job['id'],bistability='yes',offset=160,limit=80)
        self.assertEqual(n,3000)
        self.assertEqual(len(rows),80)
        selected=rows[0]
        cont=self.repo.create_job('Seed continuation',self.analyses['continuation_1d_demo'],
                   {'seed_id':selected['seed_id'],'kw':json.loads(selected['values_json'])['kw']})
        cfg=json.loads((Path(cont['output_dir'])/'config.json').read_text(encoding='utf-8'))
        self.assertEqual(cfg['seed_source_job_id'],job['id'])
        self.assertEqual(cfg['seed_values'],json.loads(selected['values_json']))
        self.assertIn('bistability',cfg['seed_metrics'])
        n,rows=self.repo.find_seeds(query='-005999',limit=80)
        self.assertEqual(n,1)
        self.assertEqual(rows[0]['seed_id'],'S-abc12345-005999')

    def test_parameter_file_autodiscovery(self):
        a=self.analyses['example_documented']
        self.assertAlmostEqual(a.parameters['mu_M']['default'],0.174)
        self.assertAlmostEqual(validate(a,{'mu_M':'0.3'})['mu_M'],0.3)

    def test_seed_override_uses_effective_values_without_mutating_source(self):
        original=self.repo.create_job('Parent',self.analyses['example_documented'],
                                      {'kw':0.5,'mu_S':0.07,'points':8})
        self.repo.update_job(original['id'],status='completed')
        self.repo.register_run_seed(original['id'])
        seed_id='RUN-'+original['id']
        job=self.repo.create_job('Child',self.analyses['continuation_1d_demo'],
                  {'kw':0.4,'points':15},seed_id=seed_id,
                  extra_parameters={'mu_S':0.09,'mu_P':1.25})
        cfg=json.loads((Path(job['output_dir'])/'config.json').read_text())
        self.assertEqual(cfg['seed_values']['mu_S'],0.07)
        self.assertEqual(cfg['parameters']['kw'],0.4)
        self.assertEqual(cfg['parameters']['mu_S'],0.09)
        self.assertEqual(cfg['parameters']['mu_P'],1.25)
        self.assertEqual(cfg['seed_overrides']['mu_S'],0.09)
        self.assertEqual(cfg['seed_overrides']['kw'],0.4)
        self.assertEqual(cfg['seed_id'],seed_id)
        self.assertEqual(self.repo.get_seed(seed_id)['source_job_id'],original['id'])

    def test_seed_can_be_used_by_analysis_without_seed_id_manifest_field(self):
        source=self.repo.create_job('Source',self.analyses['sensitivity_demo'],{'sample_count':5})
        self.repo.update_job(source['id'],status='completed')
        self.repo.register_run_seed(source['id'])
        target=self.repo.create_job('Target',self.analyses['timeseries_demo'],{'days':5},
                                   seed_id='RUN-'+source['id'],extra_parameters={'sample_count':10})
        cfg=json.loads((Path(target['output_dir'])/'config.json').read_text())
        self.assertEqual(cfg['parameters']['sample_count'],10)
        self.assertEqual(cfg['seed_source_job_id'],source['id'])

    def test_documented_example_discovered(self):
        self.assertIn('example_documented',self.analyses)
        self.assertIn('seed_id',self.analyses['example_documented'].parameters)

    def test_old_seed_database_schema_is_migrated(self):
        """Old v1 seed entries must remain available after upgrading."""
        import sqlite3
        dbpath=self.root/'legacy'/'old.sqlite'
        dbpath.parent.mkdir(parents=True,exist_ok=True)
        conn=sqlite3.connect(dbpath)
        try:
            with conn:
                conn.execute("CREATE TABLE seeds (seed_id TEXT PRIMARY KEY, source_job_id TEXT NOT NULL, values_json TEXT NOT NULL, created_at TEXT NOT NULL)")
                conn.execute("INSERT INTO seeds VALUES (?,?,?,?)", ('S-OLD','job123',json.dumps({'kw':0.3}),'2026-11-20'))
        finally:
            conn.close()
        old=Repository(dbpath,self.root/'legacy'/'runs')
        self.assertEqual(old.seed_count(),1)
        self.assertEqual(json.loads(old.get_seed('S-OLD')['values_json'])['kw'],0.3)
        self.assertIn('metrics_json',old.get_seed('S-OLD'))

    def test_sequential_claim(self):
        a=self.analyses['timeseries_demo']
        ids=[self.repo.create_job(f'run{i}',a,{'days':5,'delay_ms':0})['id'] for i in range(2)]
        self.repo.enqueue(ids)
        claimed=self.repo.claim_next()
        self.assertEqual(claimed['id'],ids[0])
        self.assertEqual(self.repo.get_job(ids[1])['status'],'queued')

if __name__=='__main__':unittest.main()
