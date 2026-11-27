import ast
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import os
import unittest
from unittest.mock import patch
import numpy as np
from model.shared_eqn import Parameters,solve_equilibria
from model.continuation_config import Axis,parse_job
from model.periodic_config import periodic_options,node_count
from model.periodic_math import (absence_certificate,deterministic_guesses,
    floquet_classification,sample_metrics,admissible_msp)
from model.periodic_orbits import equilibrium_evidence,analyze_periodic,matching_branch,run_isolated


class PeriodicTests(unittest.TestCase):
    def row(self,n=80):
        p=Parameters(N_wc=n)
        e=next(e for e in solve_equilibria(p) if e.community=='MSP')
        return p,dict(community='MSP',stability=e.stability.classification,**dict(zip('MSP',e.biomass)))

    def test_configuration_validation_and_legacy_disable(self):
        self.assertEqual(periodic_options({'oscillation_checks':'off'})['periodic_analysis'],'off')
        options=periodic_options({'periodic_nodes':31,'periodic_order':3})
        self.assertEqual(node_count(options),36)
        self.assertEqual(periodic_options(options),options)
        for values in ({'periodic_amplitudes':'nan'}, {'periodic_min_period':100,'periodic_max_period':10},
                       {'periodic_samples':3.5},{'periodic_mode':'BDF2'}):
            with self.assertRaises(ValueError):periodic_options(values)
        self.assertEqual(parse_job({'oscillation_checks':'off'},1)[3]['periodic_analysis'],'off')

    def test_absence_is_not_a_failed_search(self):
        result=equilibrium_evidence(Parameters(I_0=0))
        self.assertEqual(result['status'],'msp_absent_certified')
        p,_=self.row()
        with patch('model.periodic_orbits.solve_equilibria',return_value=[]):
            self.assertEqual(equilibrium_evidence(p)['status'],'msp_not_found')
        with patch('model.periodic_orbits.solve_equilibria',side_effect=RuntimeError('failed')):
            self.assertEqual(equilibrium_evidence(p)['status'],'equilibrium_discovery_failed')

    def test_exclusion_does_not_reject_known_msp(self):
        for n in (25,40,65,80,100,120,150):
            p=Parameters(N_wc=n)
            if any(e.community=='MSP' for e in solve_equilibria(p)):
                self.assertIsNone(absence_certificate(p),n)

    def test_stable_msp_is_included_and_reused(self):
        p,row=self.row(100)
        self.assertEqual(row['stability'],'stable')
        with patch('model.periodic_orbits.solve_equilibria',side_effect=AssertionError('must reuse')):
            result=equilibrium_evidence(p,[row])
        self.assertEqual(result['status'],'msp_present')
        self.assertTrue(admissible_msp(row,p))
        self.assertFalse(admissible_msp(dict(row,community='MP'),p))

    def test_deterministic_guesses_include_real_spectra(self):
        p,row=self.row();options=periodic_options({})
        first=list(deterministic_guesses(row,p,options))
        self.assertEqual(first,list(deterministic_guesses(row,p,options)))
        self.assertTrue(first[0]['mode'].startswith('complex'))
        self.assertEqual({g['amplitude'] for g in first},set(options['periodic_amplitudes']))
        with patch('model.periodic_math.jacobian',return_value=np.diag([-1.,-2.,-3.])):
            real=list(deterministic_guesses(row,p,options))
        self.assertTrue(real)
        self.assertTrue(all(g['mode'].startswith('real_plane') for g in real))

    def test_floquet_removes_only_one_unity_multiplier(self):
        self.assertEqual(floquet_classification([1,.8,.2],1e-4)['stability'],'stable')
        self.assertEqual(floquet_classification([1,1.01,.2],1e-4)['stability'],'unstable')
        self.assertEqual(floquet_classification([1,1,.2],1e-4)['stability'],'nonhyperbolic')
        self.assertEqual(floquet_classification([.9,.8,.2],1e-4)['stability'],'unresolved')
        self.assertEqual(floquet_classification([1,.8],1e-4)['stability'],'unresolved')

    def test_stationary_and_nonphysical_curves_rejected(self):
        p,row=self.row();state=np.array([row[c] for c in 'MSP'])
        with self.assertRaisesRegex(ValueError,'Collapsed'):
            sample_metrics(np.tile(state,(128,1)),450,p,periodic_options({}))
        bad=np.tile(state,(128,1));bad[0,2]=-1
        with self.assertRaisesRegex(ValueError,'MSP-positive'):
            sample_metrics(bad,450,p,periodic_options({}))
        bad=np.tile(state,(128,1));bad[0,0]=p.N_wc
        with self.assertRaisesRegex(ValueError,'nitrogen'):
            sample_metrics(bad,450,p,periodic_options({}))

    def test_periodic_modules_have_no_time_integration_calls(self):
        root=Path(__file__).resolve().parents[1]/'model'
        for path in root.glob('periodic*.py'):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node,ast.Call):
                    name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
                    self.assertNotIn(name,('solve_ivp','odeint','run','solve_unsteady','run_transient'),str(path))
                    self.assertFalse(any(k.arg=='timestep' for k in node.keywords),str(path))

    def test_disabled_and_excluded_locations_start_no_workers(self):
        for values,base in (({'periodic_analysis':'off'},Parameters()),
                            ({'periodic_samples':3,'periodic_refinement_rounds':0},Parameters(I_0=0))):
            with tempfile.TemporaryDirectory() as folder,patch('model.periodic_orbits.run_isolated',side_effect=AssertionError):
                data={'scan':[]}
                analyze_periodic(data,base,[Axis('N_wc',1,100,100)],values,lambda _:None,folder)
                self.assertEqual(data['oscillations'],[])
                self.assertTrue(list(Path(folder).glob('periodic_H*.json')))

    def test_unsuccessful_native_search_is_not_cycle_absence(self):
        p,row=self.row(100)
        with tempfile.TemporaryDirectory() as folder,\
             patch('model.periodic_orbits.equilibrium_evidence',return_value=dict(status='msp_present',roots=[row],msp=[row])),\
             patch('model.periodic_orbits.deterministic_guesses',return_value=iter([dict(kind='manual',equilibrium=row)])),\
             patch('model.periodic_orbits.run_isolated',return_value=dict(reason='native_worker_crashed',events=[],exit_code=99,seconds=1)):
            data={'scan':[]}
            analyze_periodic(data,p,[Axis('N_wc',90,110,100)],dict(periodic_samples=3,periodic_refinement_rounds=0),lambda _:None,folder)
        self.assertTrue(all(l['status']=='orbit_discovery_unsuccessful' for l in data['periodic_analysis']['locations']))
        self.assertEqual(data['oscillations'],[])

    def test_collocation_plot_keeps_stability_and_branch_order(self):
        from scripts.continuation_plots import plt,draw_periodic_envelopes
        def point(x,stability):return dict(x=x,S_min=2,S_max=4,S_mean=3,stability=stability)
        first=point(1,'stable');middle=point(2,'stable');last=point(1.8,'unstable')
        data=dict(periodic_analysis=dict(branches=[dict(points=[first,middle,last],arms=[dict(records=[middle,last])])]))
        fig,ax=plt.subplots();draw_periodic_envelopes(ax,data,'S',True)
        self.assertEqual(len(ax.collections),2)  # isolated initial range + stable fill
        self.assertEqual(len(ax.lines),2)  # unstable min/max, no stable fill
        self.assertEqual(list(ax.lines[0].get_xdata()),[2,1.8])
        plt.close(fig)

    def test_native_unstable_cycle_continues_through_fold(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'normal_form.json'
            env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            result=subprocess.run([sys.executable,'-m','tests.fixtures.periodic_normal_form',str(output)],
                cwd=Path(__file__).resolve().parents[1],env=env,capture_output=True,text=True,timeout=120,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.assertEqual(result.returncode,0,(result.stdout+result.stderr)[-3000:])
            rows=json.loads(output.read_text())
        self.assertEqual(rows[0]['stability'],'unstable')
        self.assertEqual(rows[-1]['stability'],'stable')
        u=np.array([r['u'] for r in rows]);du=np.diff(u)
        self.assertTrue(np.any(du<0) and np.any(du>0))
        self.assertAlmostEqual(float(u.min()),.25,delta=.01)
        self.assertTrue(all(abs(r['period']-2*np.pi)<.002 for r in rows))
        self.assertTrue(all(r['residual']<1e-7 for r in rows))

    def test_worker_crash_preserves_flushed_records(self):
        real_popen=subprocess.Popen
        code=('import pathlib,sys,json,os; '
              'p=pathlib.Path(sys.argv[1]).with_suffix(".jsonl"); '
              'p.write_text(json.dumps({"type":"checkpoint","value":7})+"\\n"); os._exit(37)')
        def launch(command,**kwargs):return real_popen([sys.executable,'-c',code,command[-1]],**kwargs)
        with tempfile.TemporaryDirectory() as folder,patch('model.periodic_orbits.subprocess.Popen',side_effect=launch):
            seen=[]
            result=run_isolated(dict(kind='manual'),Parameters(),[Axis('N_wc',1,100,100)],
                                periodic_options({}),folder,lambda _:None,seen.append)
        self.assertEqual(result['reason'],'native_worker_crashed')
        self.assertEqual(result['exit_code'],37)
        self.assertEqual(seen,[dict(type='checkpoint',value=7)])

    def test_worker_timeout_does_not_escape_supervisor(self):
        real_popen=subprocess.Popen
        def launch(command,**kwargs):return real_popen([sys.executable,'-c','import time; time.sleep(10)'],**kwargs)
        options=periodic_options({});options['periodic_timeout']=.2
        with tempfile.TemporaryDirectory() as folder,patch('model.periodic_orbits.subprocess.Popen',side_effect=launch):
            result=run_isolated(dict(kind='manual'),Parameters(),[Axis('N_wc',1,100,100)],options,folder,lambda _:None)
        self.assertEqual(result['reason'],'worker_timeout')
