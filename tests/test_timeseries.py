import tempfile
import unittest
import numpy as np
from scipy.integrate import solve_ivp
from app.core.control import RunContext
from app.core.discovery import discover
from model.timeseries_config import parse_job
from model.shared_eqn import rhs
from scripts.timeseries_runner import integrate, run


class TimeSeriesTests(unittest.TestCase):
    def test_discovery_and_validation(self):
        self.assertEqual(set(discover()),{'timeseries','continuation_1d','continuation_2d','sensitivity'})
        p,depths,groups,initial,v=parse_job({'plot_groups':'MS, S, P','depth_count':'2','depths':'2, 4'})
        self.assertEqual(groups,['MS','S','P'])
        for invalid in ({'plot_groups':'MS, X'},{'depths':'0'},{'initial_P':100},{'plot_groups':'MM'}):
            with self.assertRaises(ValueError):parse_job(invalid)

    def test_integration_matches_shared_rhs_and_inventory(self):
        p,_,_,initial,_=parse_job({})
        t,y=integrate(p,initial,30,51)
        reference=solve_ivp(rhs,(0,30),[1,1,1/p.H],args=(p,),t_eval=t,rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(y[:3],reference.y*np.array([[1],[1],[p.H]]),rtol=2e-5,atol=1e-7)
        np.testing.assert_allclose(y[0]+y[2]+y[3],p.N_wc)
        np.testing.assert_allclose(y[:3,0],initial)

    def test_grouped_depth_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=RunContext(tmp)
            run({'duration':2,'samples':9,'depth_count':'2','depths':'2,4','plot_groups':'MS, S, P'},ctx)
            self.assertEqual(len(list((ctx.folder/'figures').glob('*.png'))),3)
            self.assertEqual(len(list((ctx.folder/'figures').glob('*.pdf'))),3)
            for h in (2,4):
                data=np.loadtxt(ctx.folder/f'timeseries_H{h}.csv',delimiter=',',skiprows=1)
                self.assertEqual(data.shape,(9,5))
                np.testing.assert_allclose(data[0,1:4],[1,1,1])
