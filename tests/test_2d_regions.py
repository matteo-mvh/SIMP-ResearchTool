import unittest
from unittest.mock import patch
from model.shared_eqn import Parameters,solve_equilibria,per_capita
from model.continuation_regions import analyze_regions
from model.continuation_config import Axis
from scipy.optimize import brentq
import numpy as np


class RegionRegressionTests(unittest.TestCase):
    def test_native_boundary_corrector_refreshes_resident_guess(self):
        from model.pyoomph_continuation import correct_event,validate_event
        axes=[Axis('N_wc',1,2000,101),Axis('kw',.1,.7,101)]
        seed={'community':'M','u':1.,'v':.3753093993660956,'M':1988.6860964426703,'S':0.,'P':0.}
        result=correct_event(Parameters(H=6),axes,seed,'invasion','P',1)
        self.assertAlmostEqual(result['u'],1.)
        self.assertAlmostEqual(result['v'],.3728084114370523,places=6)
        self.assertTrue(validate_event(result,Parameters(H=6),axes,'invasion','P'))

    def test_legend_excludes_invisible_slivers(self):
        from shapely.geometry import box
        from scripts.continuation_plots import depth_axes,visible_region_labels,plt
        fig,axes=depth_axes(1)
        data=[{'regions':[{'polygon':box(0,0,.7,1),'label':'M+P'},
                           {'polygon':box(.7,0,1,1),'label':'P'},
                           {'polygon':box(0,0,1e-5,1),'label':'S'}]}]
        self.assertEqual(visible_region_labels(data,axes),{'M+P','P'})
        plt.close(fig)

    def test_large_inventory_phytoplankton_root(self):
        for h,n,k in ((4,1818.3085432756768,.5923804145373885),
                      (6,1819.6429349457026,.37545952852872755),(8,1766.0906010945232,.31234563058838255)):
            p=Parameters(H=h,N_wc=n,kw=k)
            direct=brentq(lambda P:per_capita((0,0,P),p)[2],0,n/h,xtol=1e-13)
            found=[e for e in solve_equilibria(p) if e.community=='P']
            self.assertEqual(len(found),1)
            self.assertAlmostEqual(found[0].biomass[2],direct,places=8)
            self.assertEqual(found[0].stability.classification,'stable')

    def test_joined_endpoints_are_not_dangling(self):
        curves=[[{'u':u,'v':v} for u,v in coords] for coords in
                (((0,.5),(.5,.5),(1,.5)),((.5,0),(.5,.5)))]
        data={'curves':curves}
        root={'community':'M','stability':'stable','M':1.,'S':0.,'P':0.}
        with patch('model.continuation_regions.discover',return_value=[root]):
            analyze_regions(data,Parameters(),[Axis('N_wc',1,100,100),Axis('kw',.1,1,100)],3,lambda _:None)
        self.assertEqual(len(data['regions']),3)
        self.assertTrue(all(r['label']=='M' for r in data['regions']))

    def test_coincident_curves_do_not_generate_slivers(self):
        # Retrace suppression now compares resident states as well as geometry.
        def row(u,v):
            return dict(u=u,v=v,community='M',kind='fold',invader='',M=1.,S=0.,P=0.)
        data={'curves':[[row(0,.5),row(1,.5)],
                        [row(0,.5+1e-11),row(1,.5+1e-11)]]}
        with patch('model.continuation_regions.discover',return_value=[]):
            analyze_regions(data,Parameters(),[],3,lambda _:None)
        self.assertEqual(len(data['regions']),2)
        self.assertAlmostEqual(sum(r['polygon'].area for r in data['regions']),1)
