"""Independent numerical checks of the thesis implementation."""
from dataclasses import replace
import unittest
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from model.shared_eqn import (
    Parameters, grazing, grazing_derivative, rhs, jacobian, resources,
    per_capita, solve_equilibria, scalar_roots, classify_stability,
)


class ModelTests(unittest.TestCase):
    def test_analytical_jacobian_against_central_differences(self):
        for gamma, beta in ((0, 1), (1, 1), (1, 2)):
            p = Parameters(gamma=gamma, beta=beta, N_wc=200)
            for y in (np.array([10., 70., 3.]), np.array([0., 50., 0.]), np.zeros(3)):
                h = 1e-4
                numeric = np.column_stack([
                    (rhs(0, y+h*e, p)-rhs(0, y-h*e, p))/(2*h)
                    for e in np.eye(3)])
                np.testing.assert_allclose(jacobian(y, p), numeric, atol=2e-10, rtol=1e-7)

    def test_grazing_normalization_and_origin_derivative(self):
        for gamma, beta in ((0, 1), (1, 1), (1, 2)):
            p = Parameters(gamma=gamma, beta=beta)
            self.assertEqual(grazing(0, p), 0)
            self.assertAlmostEqual(grazing(p.K_G_S, p), p.g_M/2)
            expected = p.g_M/(2*p.K_G_S) if gamma == 0 else p.g_M/p.K_G_S if beta == 1 else 0
            self.assertAlmostEqual(grazing_derivative(0, p), expected)

    def test_reference_equilibria_and_invasion_rates(self):
        p = Parameters()
        equilibria = solve_equilibria(p)
        self.assertTrue({'0', 'M', 'S', 'P', 'SP'}.issubset({e.community for e in equilibria}))
        for e in equilibria:
            y = np.array(e.biomass)
            np.testing.assert_allclose(rhs(0, y, p), 0, atol=1e-8)
            self.assertGreaterEqual(e.nitrogen, -1e-8)
            J = jacobian(y, p)
            for i in np.flatnonzero(y == 0):
                expected = np.zeros(3)
                expected[i] = per_capita(y, p)[i]
                np.testing.assert_allclose(J[i], expected)
        bare = next(e for e in equilibria if e.community == '0')
        self.assertEqual(bare.stability.classification, 'unstable')

    def test_analytic_monoculture_without_biomass_shading(self):
        p = Parameters(k_M=0, k_P=0)
        found = {e.community: e for e in solve_equilibria(p)}
        I, _ = resources([0, 0, 0], p)
        nm = p.K_N_M*p.m_M/(p.mu_M*I[0]/(p.K_I_M+I[0])-p.m_M)
        np_ = p.K_N_P*(p.m_P+p.g_P)/(p.mu_P*I[2]/(p.K_I_P+I[2])-p.m_P-p.g_P)
        self.assertAlmostEqual(found['M'].biomass[0], p.N_wc-p.H*nm, places=7)
        self.assertAlmostEqual(found['P'].biomass[2], p.N_wc/p.H-np_, places=7)
        expected_s = (np.log(p.I_0*(p.mu_S-p.m_S)/(p.K_I_S*p.m_S))-p.H*p.kw)/p.k_S
        self.assertAlmostEqual(found['S'].biomass[1], expected_s, places=7)

    def test_roots_tangencies_gaps_and_poles(self):
        roots = scalar_roots(lambda x: (x-.31)**2*(x-.72), 0, 1)
        self.assertEqual(len(roots), 2)
        np.testing.assert_allclose(roots, [.31, .72], atol=1e-7)
        self.assertEqual(scalar_roots(lambda x: float('nan') if .4 < x < .6 else x-.5, 0, 1), [])
        self.assertEqual(scalar_roots(lambda x: 1/(x-.513), 0, 1), [])

    def test_darkness_zero_inventory_and_invalid_parameters(self):
        self.assertEqual([e.community for e in solve_equilibria(Parameters(I_0=0))], ['0'])
        self.assertEqual({e.community for e in solve_equilibria(Parameters(N_wc=0))}, {'0', 'S'})
        for kwargs in ({'H': 0}, {'gamma': 0, 'beta': 2}, {'kw': -1}, {'mu_M': float('nan')}):
            with self.assertRaises(ValueError):
                Parameters(**kwargs)
        with self.assertRaises(ValueError):
            solve_equilibria(Parameters(k_S=0))
        with self.assertRaises(ValueError):
            Parameters.from_mapping({'typo': 1})

    def test_nonhyperbolic_threshold(self):
        p = Parameters()
        I, _ = resources([0, 0, 0], p)
        p = replace(p, mu_M=0, mu_P=0, m_S=p.mu_S*I[1]/(p.K_I_S+I[1]))
        result = classify_stability([0, 0, 0], p)
        self.assertEqual(result.classification, 'nonhyperbolic')
        self.assertTrue(result.near_zero)

    def test_time_series_physical_domain_and_equilibrium_stationarity(self):
        p = Parameters()
        result = solve_ivp(rhs, (0, 500), (2, 30, 1), args=(p,), rtol=1e-9, atol=1e-11)
        self.assertTrue(result.success)
        self.assertTrue(np.all(result.y >= -1e-9))
        N = (p.N_wc-result.y[0])/p.H-result.y[2]
        self.assertTrue(np.all(N >= -1e-9))
        for e in solve_equilibria(p):
            if e.stability.classification == 'stable':
                stationary = solve_ivp(rhs, (0, 50), e.biomass, args=(p,), rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(stationary.y[:, -1], e.biomass, atol=1e-7)

    def test_reduced_search_against_independent_multistart(self):
        # Solve resident per-capita equations directly, without analytical eliminations.
        rng = np.random.default_rng(42)
        seen = set()
        for inventory, gamma, beta in ((200, 0, 1), (200, 1, 1), (200, 1, 2), (20, 1, 2)):
            p = Parameters(N_wc=inventory, g_M=.08, gamma=gamma, beta=beta)
            reduced = solve_equilibria(p)
            for label in ('M', 'S', 'P', 'MS', 'MP', 'SP', 'MSP'):
                indices = [i for i, c in enumerate('MSP') if c in label]
                def expand(x):
                    y = np.zeros(3)
                    y[indices] = x
                    return y
                def f(x):
                    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                        return per_capita(expand(x), p)[indices] if np.all(np.isfinite(x)) else np.full(len(indices), 1e30)
                for _ in range(12):
                    guess = rng.uniform(.01, 1, 3)*[.9*p.N_wc, 200, .8*p.N_wc/p.H]
                    solved = root(f, guess[indices])
                    y = expand(solved.x)
                    if (solved.success and np.all(solved.x > 1e-7)
                            and y[0]+p.H*y[2] <= p.N_wc and np.max(np.abs(f(solved.x))) < 1e-8):
                        seen.add(label)
                        self.assertTrue(any(e.community == label and np.allclose(e.biomass, y, atol=1e-5)
                                            for e in reduced), (gamma, beta, label, y))
        self.assertTrue({'M', 'S', 'P', 'MS', 'MP', 'SP', 'MSP'}.issubset(seen), seen)


if __name__ == '__main__':
    unittest.main()
