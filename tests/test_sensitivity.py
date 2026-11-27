import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import asdict
import numpy as np
from model.shared_eqn import Parameters,solve_equilibria
from model.sensitivity_config import parse_job
from scripts.sensitivity_runner import lhs_file,evaluate
from unittest.mock import patch
from types import SimpleNamespace


class SensitivityTests(unittest.TestCase):
    def test_marginal_and_failed_samples_are_not_false_negatives(self):
        eq=SimpleNamespace(stability=SimpleNamespace(classification='nonhyperbolic'))
        with patch('scripts.sensitivity_runner.solve_equilibria',return_value=[eq]):
            result=evaluate(1,asdict(Parameters()),257)
            self.assertEqual(result[2],'marginal');self.assertIsNone(result[3]);self.assertIsNone(result[4])
        with patch('scripts.sensitivity_runner.solve_equilibria',side_effect=ValueError('invalid')):
            result=evaluate(1,asdict(Parameters()),257)
            self.assertEqual(result[2],'error');self.assertIsNone(result[3])

    def test_all_fixed_lhs(self):
        with tempfile.TemporaryDirectory() as tmp:
            x=lhs_file(Path(tmp)/'fixed.npy',2,{},42)
            self.assertEqual(x.shape,(2,0));del x

    def test_lhs_strata_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            ranges={'kw':[.1,1.],'N_wc':[10,200]}
            a=lhs_file(Path(tmp)/'a.npy',50,ranges,42)
            b=lhs_file(Path(tmp)/'b.npy',50,ranges,42)
            np.testing.assert_array_equal(a,b)
            for j,(lo,hi) in enumerate(ranges.values()):
                np.testing.assert_array_equal(np.sort(np.floor((a[:,j]-lo)/(hi-lo)*50)),np.arange(50))
            del a,b

    def test_invalid_ranges_and_fixed_values(self):
        for values in ({'ranges':'{"H":[0,4]}'},{'ranges':'{"kw":[2,1]}'},{'ranges':'{"gamma":[0,1]}'},
                       {'ranges':'{}','m_S':0}):
            with self.assertRaises(ValueError):parse_job(values)
        p,r,v=parse_job({'ranges':'{}'})
        self.assertEqual(p,Parameters());self.assertEqual(r,{})

    def test_metrics_match_shared_equilibrium_solver(self):
        p=Parameters(N_wc=100,kw=.2)
        result=evaluate(0,asdict(p),257)
        stable=[e for e in solve_equilibria(p) if e.stability.classification=='stable']
        self.assertEqual(result[5],len(stable))
        self.assertEqual(result[3],int(len(stable)>=2))
        if len(stable)>=2:
            self.assertAlmostEqual(result[4],max(e.biomass[1] for e in stable)-min(e.biomass[1] for e in stable))
        else:self.assertIsNone(result[4])
        for state,e in zip(result[7],stable):self.assertAlmostEqual(state['P_area'],e.biomass[2]*p.H)
