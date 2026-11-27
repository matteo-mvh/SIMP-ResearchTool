from dataclasses import asdict,replace
import unittest
import numpy as np
from app.core.discovery import discover,validate
from model.shared_eqn import Parameters,solve_equilibria,per_capita,jacobian
from model.continuation_config import Axis,parse_job,job_schema


class ContinuationTests(unittest.TestCase):
    def test_manifests_and_layouts(self):
        found=discover()
        for dimensions in (1,2):
            spec=found[f'continuation_{dimensions}d']
            for depths in ('4','2, 4','2, 4, 6, 8'):
                values=validate(spec,{'depths':depths,'depth_count':str(len(depths.split(',')))})
                p,ds,axes,_=parse_job(values,dimensions)
                self.assertEqual(len(ds),len(depths.split(',')))
                self.assertNotIn('H',spec.parameters['x_parameter']['options'])
                self.assertNotIn('beta',spec.parameters['x_parameter']['options'])
        with self.assertRaises(ValueError):parse_job({'depth_count':'2','depths':'4'},1)
        with self.assertRaises(ValueError):parse_job({'y_parameter':'N_wc'},2)
        with self.assertRaises(ValueError):parse_job({'x_parameter':'K_N_M','x_min':0},1)

    def test_native_stationary_step_and_eigenvalues(self):
        from model.pyoomph_continuation import native_problem,point_record
        p=Parameters();axis=Axis('N_wc',10,30,20)
        eq=next(e for e in solve_equilibria(p) if e.community=='SP')
        with native_problem(p,[axis],[.5],eq.biomass,'SP') as native:
            state,uv=native.vector()
            np.testing.assert_allclose(state,eq.biomass,atol=1e-7)
            eig,_=native.solve_eigenproblem(2)
            expected=np.linalg.eigvals(jacobian(state,p)[1:,1:])
            np.testing.assert_allclose(np.sort_complex(eig),np.sort_complex(expected),atol=1e-8)
            native.arclength_continuation(native.controls[0],.01,max_ds=.05)
            state,uv=native.vector()
            self.assertGreater(uv[0],.5)
            row=point_record(p,[axis],uv,state,'SP')
            self.assertLess(row['residual'],1e-8)

    def test_native_invasion_curve(self):
        from model.pyoomph_continuation import correct_event,trace_event,point_record
        # Bare S invasion threshold is analytic: I0 exp(-H kw)=KI,S*mS/(muS-mS).
        p=Parameters();axes=[Axis('I_0',10,500,12),Axis('kw',.1,1,12)]
        controls=[.3,.5]
        seed=point_record(p,axes,controls,np.zeros(3),'0')
        event=correct_event(p,axes,seed,'invasion','S',0)
        expected=p.K_I_S*p.m_S/(p.mu_S-p.m_S)*np.exp(p.H*axes[1].physical(.5))
        self.assertAlmostEqual(event['x'],expected,places=6)
        curve,notes=trace_event(p,axes,event,80,lambda _:None)
        self.assertGreater(len(curve),5,notes)
        for row in curve:
            actual=replace(p,I_0=row['x'],kw=row['y'])
            self.assertLess(abs(per_capita([0,0,0],actual)[1]),1e-7)

    def test_native_fold_and_hopf_trackers_on_known_normal_forms(self):
        from unittest.mock import patch
        from model.pyoomph_continuation import native_problem
        axis=Axis('N_wc',20,30,20)
        def fold(M,S,P,p,exp):
            return (p.N_wc-25-(M-2)**2, -1, -1)
        with patch('model.pyoomph_continuation.growth_expressions',fold):
            with native_problem(Parameters(),[axis],[.55],[2.5,0,0],'M','fold') as native:
                state,uv=native.vector()
                self.assertAlmostEqual(state[0],2,places=6)
                self.assertAlmostEqual(axis.physical(uv[0]),25,places=6)
        def hopf(M,S,P,p,exp):
            a=p.N_wc-25
            return ((a*(M-1)-(S-1))/M, ((M-1)+a*(S-1))/S, -1)
        with patch('model.pyoomph_continuation.growth_expressions',hopf):
            with native_problem(Parameters(),[axis],[.55],[1,1,0],'MS','hopf') as native:
                state,uv=native.vector()
                np.testing.assert_allclose(state[:2],[1,1],atol=1e-7)
                self.assertAlmostEqual(axis.physical(uv[0]),25,places=6)


if __name__=='__main__':unittest.main()
