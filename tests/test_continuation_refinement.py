import unittest
from unittest.mock import patch
import numpy as np
from model.continuation_config import Axis
from model.shared_eqn import Parameters
from model.continuation_refinement import refine_regions,local_transect


class RefinementTests(unittest.TestCase):
    def test_collinear_biomass_checks_do_not_abort_plotting(self):
        from shapely.geometry import box
        from scripts.continuation_plots import plot_2d,plt
        selected=dict(M=1.,S=0.,P_area=0.,N_area=2.)
        data=[dict(H=4,curves=[],regions=[dict(label='M',polygon=box(0,0,1,1),
              samples=[dict(u=u,v=2*u,selected=selected) for u in (.1,.2,.3)])])]
        names=[]
        def save(fig,folder,name):names.append(name);plt.close(fig)
        with patch('scripts.continuation_plots.save',side_effect=save):
            plot_2d(data,[Axis('N_wc',1,100,100),Axis('kw',.1,1,100)],'unused')
        self.assertEqual(len(names),6)

    def roots(self,base,axes,uv,**kwargs):
        u,v=uv
        return [dict(community='M',stability='stable' if u<.5 else 'unstable',M=1.,S=0.,P=0.,u=u,v=v),
                dict(community='P',stability='unstable' if u<.5 else 'stable',M=0.,S=0.,P=1.,u=u,v=v)]

    def test_conflict_drives_new_curve_and_region_recheck(self):
        axes=[Axis('N_wc',1,100,100),Axis('kw',.1,1,100)]
        event=dict(community='M',kind='invasion',invader='P',solved_axis=0,u=.5,v=.5,M=1.,S=0.,P=0.)
        curve=[dict(event,v=v) for v in (0.,.5,1.)]
        def locate(base,axes,rows,axis,checkpoint,**kwargs):
            # The native corrector must always receive an axis-aligned slice.
            held=[r['uv'[1-axis]] for group in rows for r in group]
            self.assertLess(np.ptp(held),1e-10)
            # Current discovery streams prepared events through this callback.
            kwargs['on_prepared'](event,None)
            return ([event],[])
        data={'curves':[]}
        with patch('model.continuation_refinement.native.discover',side_effect=self.roots),\
             patch('model.continuation_refinement.native.locate_events',side_effect=locate),\
             patch('model.continuation_refinement.native.correct_event',return_value=event),\
             patch('model.continuation_refinement.native.trace_event',return_value=(curve,[])) as trace:
            refine_regions(data,Parameters(),axes,3,50,max_rounds=3,searches_per_round=4)
        self.assertEqual({r['label'] for r in data['regions']},{'M','P'})
        self.assertEqual(data['refinement']['stop_reason'],'resolved')
        self.assertEqual(trace.call_count,1)
        self.assertGreater(data['refinement']['rounds'][0]['unresolved_area_before'],.99)
        self.assertEqual(data['refinement']['rounds'][0]['unresolved_area_after'],0)

    def test_failed_searches_stay_unresolved_and_are_bounded(self):
        data={'curves':[]};axes=[Axis('N_wc',1,100,100),Axis('kw',.1,1,100)]
        with patch('model.continuation_refinement.native.discover',side_effect=self.roots),\
             patch('model.continuation_refinement.native.locate_events',return_value=([],[])) as locate:
            refine_regions(data,Parameters(),axes,3,50,max_rounds=3,searches_per_round=2)
        self.assertTrue(any(r['label']=='?' for r in data['regions']))
        self.assertLessEqual(locate.call_count,6)
        self.assertEqual(data['refinement']['stop_reason'],'stalled')

    def test_local_bracket_refinement_has_budget(self):
        calls=[]
        def lookup(uv):calls.append(uv);return self.roots(None,None,uv)
        rows=local_transect(np.array([0.,.3]),np.array([1.,.3]),0,lookup,max_checks=12)
        self.assertLessEqual(len(calls),12)
        self.assertTrue(all(abs(r['v']-.3)<1e-12 for group in rows for r in group))
