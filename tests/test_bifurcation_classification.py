
"""Mathematical regression tests for SIMP bifurcation classification."""

import unittest
from dataclasses import replace

import numpy as np
from scipy.optimize import brentq, minimize_scalar, root

from model.shared_eqn import (
    Parameters,
    rhs,
    jacobian,
    per_capita,
    solve_equilibria,
    resources,
    grazing,
    seagrass_biomass,
)

from model.continuation_config import Axis

from scripts.bifurcation_classification import (
    classify_local,
    classify_event,
    correct_candidate,
    audit_1d,
    critical_eigenvectors,
    state_tensors,
)


# ==============================================================
# Helpers
# ==============================================================

def event_row(state, axis, value, community, kind, invader=''):
    """Create an event record compatible with the classifier."""

    return dict(
        M=float(state[0]),
        S=float(state[1]),
        P=float(state[2]),
        u=float(axis.normalized(value)),
        x=float(value),
        community=community,
        kind=kind,
        invader=invader,
    )


def check_equilibrium(state, p, tolerance=1e-7):
    """Independent stationarity and physical-domain checks."""

    state = np.asarray(state, float)

    if np.max(abs(rhs(0., state, p))) > tolerance:
        raise AssertionError('State is not an equilibrium')

    if np.any(state < 0):
        raise AssertionError('Negative biomass')

    if resources(state, p)[1] < -tolerance:
        raise AssertionError('Negative dissolved nitrogen')


# ==============================================================
# 1. Analytical normal-form tests
# ==============================================================

class NormalFormTests(unittest.TestCase):

    def setUp(self):
        self.uv = np.array([.5])

    def assert_verified(self, result, expected):
        self.assertEqual(
            result['classification_status'],
            'verified',
            result,
        )

        self.assertEqual(
            result['classification'],
            expected,
            result,
        )

    def test_saddle_node(self):
        """x' = 2 mu - 3 x²."""

        def f(z, t):
            x, y, w = z
            mu = t[0]-.5

            return np.array([
                2*mu-3*x*x,
                -y,
                -2*w,
            ])

        def j(z, t):
            return np.diag([
                -6*z[0], -1., -2.
            ])

        result = classify_local(
            f, j, self.uv, 'fold', [0]
        )

        self.assert_verified(result, 'saddle_node')

        self.assertAlmostEqual(
            result['parameter_coefficient'], 2., places=6
        )

        self.assertAlmostEqual(
            result['nonlinear_coefficient'], -3., places=6
        )

        self.assertEqual(
            result['branch_side'],
            'increasing_parameter',
        )

        self.assertEqual(result['zero_eigenvalues'], 1)

    def test_forward_transcritical(self):
        """x' = x(mu - 2x)."""

        def f(z, t):
            x, y, w = z

            return np.array([
                x*(t[0]-.5-2*x),
                -y,
                -w,
            ])

        def j(z, t):
            return np.diag([
                t[0]-.5-4*z[0],
                -1.,
                -1.,
            ])

        result = classify_local(
            f, j, self.uv,
            'invasion', [], invader=0,
        )

        self.assert_verified(
            result, 'transcritical_forward'
        )

        self.assertAlmostEqual(
            result['parameter_coefficient'], 1., places=6
        )

        self.assertAlmostEqual(
            result['nonlinear_coefficient'], -2., places=6
        )

        self.assertEqual(
            result['emerging_branch_critical_stability'],
            'stable',
        )

        self.assertEqual(
            result['branch_side'],
            'increasing_parameter',
        )

    def test_backward_transcritical(self):
        """x' = x(mu + 2x)."""

        def f(z, t):
            x, y, w = z

            return np.array([
                x*(t[0]-.5+2*x),
                -y,
                -w,
            ])

        def j(z, t):
            return np.diag([
                t[0]-.5+4*z[0],
                -1.,
                -1.,
            ])

        result = classify_local(
            f, j, self.uv,
            'invasion', [], invader=0,
        )

        self.assert_verified(
            result, 'transcritical_backward'
        )

        self.assertAlmostEqual(
            result['nonlinear_coefficient'], 2., places=6
        )

        self.assertEqual(
            result['emerging_branch_critical_stability'],
            'unstable',
        )

        self.assertEqual(
            result['branch_side'],
            'decreasing_parameter',
        )

    def test_transcritical_with_resident_feedback(self):
        """Verify indirect ecological feedback and branch movement.

        x' = -x + 2y + 3mu
        y' = y(2mu - 3y + x)

        Resident equilibrium: x = 3mu.

        Therefore:
            a = 2 + 3 = 5
            b = -3 + 2 = -1
        """

        def f(z, t):
            x, y, w = z
            mu = t[0]-.5

            return np.array([
                -x+2*y+3*mu,
                y*(2*mu-3*y+x),
                -w,
            ])

        def j(z, t):
            x, y, w = z
            mu = t[0]-.5

            return np.array([
                [-1., 2., 0.],
                [y, 2*mu-6*y+x, 0.],
                [0., 0., -1.],
            ])

        result = classify_local(
            f, j, self.uv,
            'invasion', [0], invader=1,
        )

        self.assert_verified(
            result, 'transcritical_forward'
        )

        self.assertAlmostEqual(
            result['parameter_coefficient'], 5., places=6
        )

        self.assertAlmostEqual(
            result['nonlinear_coefficient'], -1., places=6
        )

    def hopf_model(self, cubic=-1., transverse=-1.,
                   tangent=False, frequency=2.):
        """Known Hopf normal form with a third direction."""

        def f(z, t):
            x, y, w = z
            mu = t[0]-.5

            if tangent:
                mu = mu**2

            r2 = x*x+y*y

            return np.array([
                mu*x-frequency*y+cubic*x*r2,
                frequency*x+mu*y+cubic*y*r2,
                transverse*w,
            ])

        def j(z, t):
            x, y, w = z
            mu = t[0]-.5

            if tangent:
                mu = mu**2

            return np.array([
                [
                    mu+cubic*(3*x*x+y*y),
                    -frequency+2*cubic*x*y,
                    0.,
                ],
                [
                    frequency+2*cubic*x*y,
                    mu+cubic*(x*x+3*y*y),
                    0.,
                ],
                [0., 0., transverse],
            ])

        return f, j

    def test_supercritical_hopf(self):

        f, j = self.hopf_model(cubic=-1.)

        result = classify_local(
            f, j, self.uv, 'hopf', [0, 1]
        )

        self.assert_verified(
            result, 'hopf_supercritical'
        )

        self.assertAlmostEqual(
            result['omega'], 2., places=6
        )

        self.assertAlmostEqual(
            result['hopf_period'], np.pi, places=6
        )

        self.assertAlmostEqual(
            result['first_lyapunov_coefficient'],
            -1.,
            places=5,
        )

        self.assertEqual(
            result['emerging_orbit_stability'],
            'stable',
        )

        self.assertEqual(
            result['branch_side'],
            'increasing_parameter',
        )

    def test_subcritical_hopf(self):

        f, j = self.hopf_model(cubic=1.)

        result = classify_local(
            f, j, self.uv, 'hopf', [0, 1]
        )

        self.assert_verified(
            result, 'hopf_subcritical'
        )

        self.assertAlmostEqual(
            result['first_lyapunov_coefficient'],
            1.,
            places=5,
        )

        self.assertEqual(
            result['emerging_orbit_stability'],
            'unstable',
        )

        self.assertEqual(
            result['branch_side'],
            'decreasing_parameter',
        )

    def test_supercritical_hopf_with_unstable_transverse_mode(self):
        """Supercritical does not necessarily mean fully stable."""

        f, j = self.hopf_model(
            cubic=-1.,
            transverse=1.,
        )

        result = classify_local(
            f, j, self.uv, 'hopf', [0, 1]
        )

        self.assert_verified(
            result, 'hopf_supercritical'
        )

        self.assertEqual(
            result['emerging_orbit_stability'],
            'unstable',
        )

        self.assertEqual(
            result['noncritical_unstable_directions'],
            1,
        )

    def test_hopf_along_moving_equilibrium(self):
        """Crossing speed must include equilibrium displacement.

        w' = -w + 2mu

        The radial growth rate is mu + 2w.
        Along the equilibrium w = 2mu, it becomes 5mu.
        """

        def f(z, t):
            x, y, w = z
            mu = t[0]-.5
            a = mu+2*w
            r2 = x*x+y*y

            return np.array([
                a*x-y-x*r2,
                x+a*y-y*r2,
                -w+2*mu,
            ])

        def j(z, t):
            x, y, w = z
            a = t[0]-.5+2*w

            return np.array([
                [
                    a-3*x*x-y*y,
                    -1-2*x*y,
                    2*x,
                ],
                [
                    1-2*x*y,
                    a-x*x-3*y*y,
                    2*y,
                ],
                [0., 0., -1.],
            ])

        result = classify_local(
            f, j, self.uv, 'hopf', [0, 1, 2]
        )

        self.assert_verified(
            result, 'hopf_supercritical'
        )

        self.assertAlmostEqual(
            result['parameter_coefficient'], 5., places=6
        )

        self.assertAlmostEqual(
            result['first_lyapunov_coefficient'],
            -2.,
            places=5,
        )


# ==============================================================
# 2. Reject false and degenerate bifurcations
# ==============================================================

class DegeneracyTests(unittest.TestCase):

    def setUp(self):
        self.uv = np.array([.5])

    def assert_unresolved(self, result):

        self.assertEqual(
            result['classification'],
            'unresolved',
            result,
        )

        self.assertEqual(
            result['classification_status'],
            'unresolved',
            result,
        )

        self.assertTrue(
            result['classification_reason']
        )

    def test_zero_eigenvalue_without_saddle_node(self):
        """A zero eigenvalue alone is insufficient.

        x' = mu - x³
        Quadratic coefficient is zero.
        """

        def f(z, t):
            x, y, w = z

            return np.array([
                t[0]-.5-x**3,
                -y,
                -w,
            ])

        def j(z, t):
            return np.diag([
                -3*z[0]**2,
                -1.,
                -1.,
            ])

        self.assert_unresolved(
            classify_local(
                f, j, self.uv, 'fold', [0]
            )
        )

    def test_degenerate_transcritical(self):
        """x' = x(mu - x²) has zero quadratic coefficient."""

        def f(z, t):
            x, y, w = z

            return np.array([
                x*(t[0]-.5-x*x),
                -y,
                -w,
            ])

        def j(z, t):
            return np.diag([
                t[0]-.5-3*z[0]**2,
                -1.,
                -1.,
            ])

        self.assert_unresolved(
            classify_local(
                f, j, self.uv,
                'invasion', [], invader=0,
            )
        )

    def test_tangential_invasion_threshold(self):
        """The invasion rate touches zero but does not cross it."""

        def f(z, t):
            x, y, w = z

            return np.array([
                x*((t[0]-.5)**2-x),
                -y,
                -w,
            ])

        def j(z, t):
            return np.diag([
                (t[0]-.5)**2-2*z[0],
                -1.,
                -1.,
            ])

        self.assert_unresolved(
            classify_local(
                f, j, self.uv,
                'invasion', [], invader=0,
            )
        )

    def test_double_zero_eigenvalue(self):
        """Two critical zero eigenvalues are not a generic fold."""

        def f(z, t):
            x, y, w = z

            return np.array([
                t[0]-.5+x*x,
                y*y,
                -w,
            ])

        def j(z, t):
            return np.diag([
                2*z[0],
                2*z[1],
                -1.,
            ])

        self.assert_unresolved(
            classify_local(
                f, j, self.uv, 'fold', [0, 1]
            )
        )

    def test_zero_hopf_interaction(self):
        """Imaginary pair plus zero eigenvalue is not generic Hopf."""

        f, j = NormalFormTests().hopf_model(
            cubic=-1.,
            transverse=0.,
        )

        self.assert_unresolved(
            classify_local(
                f, j, self.uv, 'hopf', [0, 1, 2]
            )
        )

    def test_degenerate_hopf(self):
        """Zero first Lyapunov coefficient requires further analysis."""

        f, j = NormalFormTests().hopf_model(
            cubic=0.
        )

        self.assert_unresolved(
            classify_local(
                f, j, self.uv, 'hopf', [0, 1]
            )
        )

    def test_tangential_hopf(self):
        """An imaginary pair must cross the imaginary axis."""

        f, j = NormalFormTests().hopf_model(
            cubic=-1.,
            tangent=True,
        )

        self.assert_unresolved(
            classify_local(
                f, j, self.uv, 'hopf', [0, 1]
            )
        )

    def test_damped_oscillation_is_not_hopf(self):

        f, j = NormalFormTests().hopf_model()

        # Negative real part: damped oscillation.
        self.assert_unresolved(
            classify_local(
                f, j, np.array([.4]),
                'hopf', [0, 1],
            )
        )

    def test_nonstationary_point_is_rejected(self):

        def f(z, t):
            return np.array([1., 0., 0.])

        def j(z, t):
            return np.diag([0., -1., -1.])

        self.assert_unresolved(
            classify_local(
                f, j, self.uv, 'fold', [0]
            )
        )


# ==============================================================
# 3. Tests using the actual SIMP equations
# ==============================================================

class SIMPBifurcationTests(unittest.TestCase):

    def test_seagrass_transcritical_threshold(self):
        """Analytical invasion threshold between 0 and S."""

        p = Parameters(N_wc=.2)

        critical_light = (
            p.K_I_S*p.m_S/(p.mu_S-p.m_S)
            * np.exp(p.H*p.kw)
        )

        axis = Axis(
            'I_0',
            .5*critical_light,
            1.5*critical_light,
            100,
        )

        row = event_row(
            [0., 0., 0.],
            axis,
            critical_light,
            '0',
            'invasion',
            'S',
        )

        result = classify_event(row, p, [axis])

        self.assertEqual(
            result['classification'],
            'transcritical_forward',
            result,
        )

        self.assertEqual(
            result['emerging_community'], 'S'
        )

        self.assertEqual(
            result['resident_community'], '0'
        )

        # Expected analytical coefficients.
        a_expected = (
            p.m_S*(1-p.m_S/p.mu_S)
        )

        b_expected = -p.k_S*a_expected

        np.testing.assert_allclose(
            result['parameter_coefficient'],
            a_expected,
            rtol=1e-5,
        )

        np.testing.assert_allclose(
            result['nonlinear_coefficient'],
            b_expected,
            rtol=1e-5,
        )

        # Independently verify equilibrium existence and stability
        # on either side of the threshold.
        below = replace(
            p, I_0=.99*critical_light
        )

        above = replace(
            p, I_0=1.01*critical_light
        )

        self.assertLess(
            per_capita([0., 0., 0.], below)[1], 0.
        )

        self.assertGreater(
            per_capita([0., 0., 0.], above)[1], 0.
        )

        roots_below = solve_equilibria(below)
        roots_above = solve_equilibria(above)

        self.assertFalse(
            any(e.community == 'S' for e in roots_below)
        )

        seagrass = next(
            e for e in roots_above
            if e.community == 'S'
        )

        expected_biomass = np.log(1.01)/p.k_S

        self.assertAlmostEqual(
            seagrass.biomass[1],
            expected_biomass,
            places=6,
        )

        self.assertEqual(
            seagrass.stability.classification,
            'stable',
        )

    def ms_fixture(self):
        """Parameters supporting backward invasion and an MS fold."""

        return Parameters(
            N_wc=200.,
            g_M=.10,
            g_P=.90,
        )

    def macroalgae_net_production(self, M, p):
        """Macroalgal growth before background mortality.

        S is constrained to its analytical equilibrium.
        """

        S = seagrass_biomass(M, 0., p)

        return float(
            per_capita([M, S, 0.], p)[0]+p.m_M
        )

    def test_backward_macroalgae_transcritical(self):
        """Actual SIMP threshold between S and MS."""

        p = self.ms_fixture()

        S = seagrass_biomass(0., 0., p)

        critical_mortality = (
            self.macroalgae_net_production(0., p)
        )

        axis = Axis(
            'm_M',
            .5*critical_mortality,
            1.5*critical_mortality,
            100,
        )

        row = event_row(
            [0., S, 0.],
            axis,
            critical_mortality,
            'S',
            'invasion',
            'M',
        )

        result = classify_event(row, p, [axis])

        self.assertEqual(
            result['classification'],
            'transcritical_backward',
            result,
        )

        self.assertEqual(
            result['emerging_community'], 'MS'
        )

        self.assertGreater(
            result['nonlinear_coefficient'], 0.
        )

        self.assertLess(
            result['parameter_coefficient'], 0.
        )

        # Increasing macroalgal mortality makes invasion impossible,
        # but a positive unstable MS equilibrium exists locally.
        test_p = replace(
            p, m_M=critical_mortality+1e-6
        )

        self.assertLess(
            per_capita([0., S, 0.], test_p)[0],
            0.,
        )

        roots = [
            e for e in solve_equilibria(test_p)
            if e.community == 'MS'
        ]

        self.assertTrue(roots)

        emerging = min(
            roots, key=lambda e: e.biomass[0]
        )

        self.assertGreater(
            emerging.biomass[0], 0.
        )

        self.assertLess(
            emerging.biomass[0], 1.
        )

        self.assertEqual(
            emerging.stability.classification,
            'unstable',
        )

    def test_simp_saddle_node(self):
        """Locate an MS saddle-node independently of PyOOMPH."""

        p = self.ms_fixture()

        optimum = minimize_scalar(
            lambda M: -self.macroalgae_net_production(M, p),
            bounds=(100., 170.),
            method='bounded',
            options={'xatol': 1e-10},
        )

        self.assertTrue(optimum.success)

        M = float(optimum.x)

        S = seagrass_biomass(M, 0., p)

        critical_mortality = (
            self.macroalgae_net_production(M, p)
        )

        actual = replace(
            p, m_M=critical_mortality
        )

        state = np.array([M, S, 0.])

        check_equilibrium(state, actual)

        axis = Axis(
            'm_M',
            critical_mortality-.005,
            critical_mortality+.005,
            100,
        )

        row = event_row(
            state,
            axis,
            critical_mortality,
            'MS',
            'fold',
        )

        result = classify_event(row, p, [axis])

        self.assertEqual(
            result['classification'],
            'saddle_node',
            result,
        )

        self.assertEqual(
            result['branch_side'],
            'decreasing_parameter',
        )

        # Independently check that two MS equilibria exist below
        # the critical mortality and disappear above it.
        below = replace(
            p, m_M=critical_mortality-1e-4
        )

        above = replace(
            p, m_M=critical_mortality+1e-4
        )

        roots_below = [
            e for e in solve_equilibria(below)
            if e.community == 'MS'
        ]

        roots_above = [
            e for e in solve_equilibria(above)
            if e.community == 'MS'
        ]

        self.assertEqual(len(roots_below), 2)
        self.assertEqual(len(roots_above), 0)

        self.assertEqual(
            {
                e.stability.classification
                for e in roots_below
            },
            {'stable', 'unstable'},
        )

    def hopf_fixture(self):
        """Construct an actual MSP equilibrium at a Hopf candidate.

        Production coefficients are adjusted to satisfy stationarity.
        Grazing is then varied to locate the imaginary-axis crossing.

        This is a numerical test fixture, not a calibrated ecological
        parameterization.
        """

        M = 6.059773352405747
        S = 10.801407858575864
        P = 1.1928066482272344
        H = 3.3737951866745792
        N = 22.1102914957444

        state = np.array([M, S, P])

        base = Parameters(
            H=H,
            N_wc=M+H*(P+N),
            k_P=.13355321071117252,
            k_M=.00030418793060679576,
            k_S=.001981957849943959,
            K_G_S=16.970567360765177,
            m_SM=4.405403046871419e-6,
        )

        def equilibrium_parameters(g_M):

            p = replace(base, g_M=float(g_M))

            light, nitrogen = resources(state, p)

            IM, IS, IP = light

            mu_M = (
                (p.m_M+grazing(S, p))
                / (
                    IM/(p.K_I_M+IM)
                    * nitrogen/(p.K_N_M+nitrogen)
                )
            )

            mu_S = (
                (p.m_S+p.m_SM*M)
                / (IS/(p.K_I_S+IS))
            )

            mu_P = (
                (p.m_P+p.g_P)
                / (
                    IP/(p.K_I_P+IP)
                    * nitrogen/(p.K_N_P+nitrogen)
                )
            )

            return replace(
                p,
                mu_M=float(mu_M),
                mu_S=float(mu_S),
                mu_P=float(mu_P),
            )

        def hopf_condition(g_M):

            p = equilibrium_parameters(g_M)

            A = jacobian(state, p)

            c = np.poly(A)

            return float(c[1]*c[2]-c[3])

        critical_grazing = brentq(
            hopf_condition,
            0.,
            .1,
            xtol=1e-14,
        )

        p = equilibrium_parameters(
            critical_grazing
        )

        return p, state

    def test_simp_hopf(self):
        """Verify a Hopf candidate in the actual three-species model."""

        p, state = self.hopf_fixture()

        check_equilibrium(state, p)

        eigenvalues = np.linalg.eigvals(
            jacobian(state, p)
        )

        pair = [
            value for value in eigenvalues
            if abs(value.imag) > 1e-6
        ]

        self.assertEqual(len(pair), 2)

        self.assertTrue(all(
            abs(value.real) < 1e-8
            for value in pair
        ))

        axis = Axis(
            'g_M',
            .5*p.g_M,
            1.5*p.g_M,
            100,
        )

        row = event_row(
            state,
            axis,
            p.g_M,
            'MSP',
            'hopf',
        )

        result = classify_event(row, p, [axis])

        self.assertEqual(
            result['classification_status'],
            'verified',
            result,
        )

        self.assertIn(
            result['classification'],
            (
                'hopf_supercritical',
                'hopf_subcritical',
            ),
        )

        self.assertAlmostEqual(
            result['omega'],
            max(value.imag for value in pair),
            places=7,
        )

        # Independently follow the equilibrium as g_M changes,
        # holding all other parameters fixed.
        def real_part_at(u):

            test_p = replace(
                p, g_M=axis.physical(u)
            )

            solution = root(
                lambda z: per_capita(np.exp(z), test_p),
                np.log(state),
                tol=1e-10,
            )

            y = np.exp(solution.x)

            self.assertLess(
                np.max(abs(per_capita(y, test_p))),
                1e-8,
            )

            eig = np.linalg.eigvals(
                jacobian(y, test_p)
            )

            mode = max(
                eig, key=lambda value: value.imag
            )

            self.assertGreater(mode.imag, 1e-6)

            return float(mode.real)

        h = 1e-4

        before = real_part_at(.5-h)
        after = real_part_at(.5+h)

        # A genuine crossing must change sign.
        self.assertLess(before*after, 0.)

        crossing = (after-before)/(2*h)

        np.testing.assert_allclose(
            result['parameter_coefficient'],
            crossing,
            rtol=.02,
            atol=1e-8,
        )


# ==============================================================
# 4. Numerical derivative and eigenvector tests
# ==============================================================

class NumericalTests(unittest.TestCase):

    def test_critical_eigenvectors(self):

        A = np.array([
            [0., -2., 0.],
            [2., 0., 0.],
            [0., 0., -1.],
        ])

        q, w = critical_eigenvectors(A, 2j)

        np.testing.assert_allclose(
            A@q, 2j*q, atol=1e-10
        )

        np.testing.assert_allclose(
            A.T@w, -2j*w, atol=1e-10
        )

        np.testing.assert_allclose(
            np.vdot(w, q), 1., atol=1e-10
        )

    def test_second_and_third_derivatives(self):
        """Known polynomial with exact Hessian and third derivative."""

        # f1 = 2*x² + 3*x*y + 4*x³
        # f2 = -y
        # f3 = -z

        def j(z):

            x, y, w = z

            return np.array([
                [4*x+3*y+12*x*x, 3*x, 0.],
                [0., -1., 0.],
                [0., 0., -1.],
            ])

        B, C = state_tensors(
            j, 1e-3, third=True
        )

        self.assertAlmostEqual(
            B[0, 0, 0], 4., places=6
        )

        self.assertAlmostEqual(
            B[0, 0, 1], 3., places=6
        )

        self.assertAlmostEqual(
            B[0, 1, 0], 3., places=6
        )

        self.assertAlmostEqual(
            C[0, 0, 0, 0], 24., places=5
        )

    def test_two_parameter_unfolding(self):
        """Select the parameter direction that unfolds the event."""

        def f(z, t):

            x, y, w = z

            return np.array([
                t[1]-.5+x*x,
                -y,
                -w,
            ])

        def j(z, t):

            return np.diag([
                2*z[0], -1., -1.
            ])

        result = classify_local(
            f,
            j,
            np.array([.5, .5]),
            'fold',
            [0],
        )

        self.assertEqual(
            result['classification'],
            'saddle_node',
            result,
        )

        self.assertEqual(
            result['classification_axis'], 1
        )

        self.assertAlmostEqual(
            result['unfolding_u'], 0., places=7
        )

        self.assertAlmostEqual(
            result['unfolding_v'], 1., places=7
        )

    def test_parameter_domain_boundary(self):
        """One-sided parameter derivatives at u=0."""

        def f(z, t):
            x, y, w = z

            return np.array([
                x*(t[0]-x),
                -y,
                -w,
            ])

        def j(z, t):

            return np.diag([
                t[0]-2*z[0], -1., -1.
            ])

        result = classify_local(
            f,
            j,
            np.array([0.]),
            'invasion',
            [],
            invader=0,
        )

        self.assertEqual(
            result['classification'],
            'transcritical_forward',
            result,
        )

        self.assertAlmostEqual(
            result['parameter_coefficient'],
            1.,
            places=6,
        )


# ==============================================================
# 5. Independent event-discovery tests
# ==============================================================

class DiscoveryTests(unittest.TestCase):

    def setUp(self):

        self.p = Parameters(N_wc=.2)

        self.threshold = (
            self.p.K_I_S*self.p.m_S
            / (self.p.mu_S-self.p.m_S)
            * np.exp(self.p.H*self.p.kw)
        )

        self.axis = Axis(
            'I_0',
            .5*self.threshold,
            1.5*self.threshold,
            100,
        )

        self.branches = [
            dict(
                branch=1,
                point=i,
                community='0',
                u=float(u),
                x=float(self.axis.physical(u)),
                M=0.,
                S=0.,
                P=0.,
            )
            for i, u in enumerate(
                np.linspace(0., 1., 20)
            )
        ]

    def test_independent_event_correction(self):

        candidate = correct_candidate(
            self.branches[9],
            self.branches[10],
            self.p,
            self.axis,
            'invasion',
            'S',
        )

        self.assertAlmostEqual(
            candidate['x'],
            self.threshold,
            places=5,
        )

        result = classify_event(
            candidate, self.p, [self.axis]
        )

        self.assertEqual(
            result['classification'],
            'transcritical_forward',
            result,
        )

    def test_missing_event_is_rediscovered(self):

        added, failures = audit_1d(
            self.branches,
            [],
            self.p,
            self.axis,
        )

        events = [
            row for row in added
            if row['invader'] == 'S'
        ]

        self.assertEqual(len(events), 1)

        self.assertEqual(
            events[0]['classification'],
            'transcritical_forward',
        )

        self.assertAlmostEqual(
            events[0]['x'],
            self.threshold,
            places=5,
        )

    def test_existing_event_is_not_duplicated(self):

        existing = event_row(
            [0., 0., 0.],
            self.axis,
            self.threshold,
            '0',
            'invasion',
            'S',
        )

        added, failures = audit_1d(
            self.branches,
            [existing],
            self.p,
            self.axis,
        )

        self.assertFalse(any(
            row['invader'] == 'S'
            for row in added
        ))


if __name__ == '__main__':
    unittest.main(verbosity=2)
