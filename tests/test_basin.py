"""Regression checks for the volume measure and conservative accounting."""
import unittest
from types import SimpleNamespace
import numpy as np
from scripts.basin_runner import sample_initial, fractions, wilson


class BasinTests(unittest.TestCase):
    def test_uniform_prism(self):
        p=SimpleNamespace(N_wc=100., H=4.)
        a=sample_initial(p,150.,20000,42)
        self.assertTrue(np.all(a>0))
        self.assertTrue(np.all(a[:,0]+p.H*a[:,2]<p.N_wc))
        self.assertTrue(np.all(a[:,1]<150.))
        np.testing.assert_allclose(a.mean(axis=0),[100/3,75,100/12],rtol=.03)
        np.testing.assert_array_equal(a,sample_initial(p,150.,20000,42))

    def test_unresolved_mass_is_not_redistributed(self):
        t=[dict(id='S',stable='stable',kind='equilibrium',color='green'),
           dict(id='M',stable='stable',kind='equilibrium',color='orange')]
        rows=fractions([0,0,1,-1],t,100.)
        self.assertEqual([r['fraction'] for r in rows],[.5,.25,.25])
        self.assertEqual(rows[0]['possible_fraction_upper'],.75)
        self.assertAlmostEqual(sum(r['sampled_volume'] for r in rows),100.)
        self.assertEqual(wilson(0,256)[0],0.)
        self.assertAlmostEqual(wilson(256,256)[1],1.)


if __name__=='__main__': unittest.main()
