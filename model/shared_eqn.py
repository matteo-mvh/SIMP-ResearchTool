"""SIMP equations and reduced equilibria (thesis 3.8-3.14, Appendix A.3).

State order is always (M, S, P); time is days. M and S are mmol N/m²,
P and dissolved N are mmol N/m³; N_wc is an areal inventory.
No GUI, filesystem, random state, or plotting dependencies live here.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from collections.abc import Mapping, Callable
import math
import numpy as np
from scipy.optimize import brentq, minimize_scalar


@dataclass(frozen=True)
class Parameters:
    # Table 3.4; environmental reference values from section 3.4.
    mu_M: float = 0.174
    mu_S: float = 0.060
    mu_P: float = 1.0
    K_I_M: float = 41.0
    K_I_S: float = 37.0
    K_I_P: float = 30.0
    K_N_M: float = 1.0
    K_N_P: float = 0.5
    k_P: float = 0.034
    k_M: float = 0.002
    k_S: float = 0.012
    m_M: float = 0.010
    m_S: float = 0.020
    m_P: float = 0.050
    g_P: float = 0.40
    g_M: float = 0.05
    K_G_S: float = 75.0
    gamma: int = 1
    beta: int = 2
    m_SM: float = 3e-5
    N_wc: float = 20.0
    H: float = 4.0
    I_0: float = 384.0
    kw: float = 0.30

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{f.name} must be finite and nonnegative')
        for name in ('H', 'K_I_M', 'K_I_S', 'K_I_P', 'K_N_M', 'K_N_P', 'K_G_S'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if (self.gamma, self.beta) not in ((0, 1), (1, 1), (1, 2)):
            raise ValueError('Grazing (gamma, beta) must be (0,1), (1,1), or (1,2)')

    @classmethod
    def from_mapping(cls, values: Mapping, *, allow_extra=False):
        """Read a script's effective parameters; extras require explicit opt-in."""
        names = {f.name for f in fields(cls)}
        unknown = set(values) - names
        if unknown and not allow_extra:
            raise ValueError(f'Unknown model parameters: {sorted(unknown)}')
        return cls(**{k: v for k, v in values.items() if k in names})


def _state(y):
    y = np.asarray(y, dtype=float)
    if y.shape != (3,) or not np.all(np.isfinite(y)):
        raise ValueError('State must contain three finite values in (M, S, P) order')
    return y


def resource_expressions(M, S, P, p, exp):
    """Single algebraic resource definition for NumPy and pyoomph expressions."""
    attenuation = p.H * (p.kw + p.k_P * P)
    return (p.I_0*exp(-attenuation-p.k_M*M/2),
            p.I_0*exp(-attenuation-p.k_M*M-p.k_S*S),
            p.I_0*exp(-attenuation/2)), (p.N_wc-M)/p.H-P


def growth_expressions(M, S, P, p, exp):
    """Backend-neutral per-capita equations used by numerical and native solvers."""
    (IM, IS, IP), N = resource_expressions(M, S, P, p, exp)
    return (p.mu_M*IM/(p.K_I_M+IM)*N/(p.K_N_M+N)-p.m_M-grazing(S, p),
            p.mu_S*IS/(p.K_I_S+IS)-p.m_S-p.m_SM*M,
            p.mu_P*IP/(p.K_I_P+IP)*N/(p.K_N_P+N)-p.m_P-p.g_P)


def resources(y, p: Parameters):
    """Return ((I_M, I_S, I_P), N), without clipping the equations."""
    light, N = resource_expressions(*_state(y), p, np.exp)
    return np.asarray(light), N


def grazing(S, p: Parameters):
    x = S / p.K_G_S
    return p.g_M * (1+p.gamma) * x**p.beta / (2*(1+p.gamma*x**p.beta))


def grazing_derivative(S, p: Parameters):
    x = S / p.K_G_S
    return (p.g_M*(1+p.gamma)*p.beta*x**(p.beta-1)
            / (2*p.K_G_S*(1+p.gamma*x**p.beta)**2))


def per_capita(y, p: Parameters):
    return np.asarray(growth_expressions(*_state(y), p, np.exp), dtype=float)


def rhs(t, y, p: Parameters):
    """ODE callback compatible with solve_ivp(..., args=(p,))."""
    y = _state(y)
    return y * np.asarray(growth_expressions(*y, p, np.exp), dtype=float)


def jacobian(y, p: Parameters):
    """Analytical full-system Jacobian, including absent-species invasion rates."""
    y = _state(y)
    I, N = resources(y, p)
    K = np.array([p.K_I_M, p.K_I_S, p.K_I_P])
    L = I/(K+I)
    dL = K*I/(K+I)**2
    dlogI = np.array([[-p.k_M/2, 0, -p.H*p.k_P],
                     [-p.k_M, -p.k_S, -p.H*p.k_P],
                     [0, 0, -p.H*p.k_P/2]])
    dN = np.array([-1/p.H, 0, -1])
    DF = np.empty((3, 3))
    for i, mu, kn in ((0, p.mu_M, p.K_N_M), (2, p.mu_P, p.K_N_P)):
        DF[i] = mu * (N/(kn+N)*dL[i]*dlogI[i] + L[i]*kn/(kn+N)**2*dN)
    DF[0, 1] -= grazing_derivative(y[1], p)
    DF[1] = p.mu_S*dL[1]*dlogI[1] - np.array([p.m_SM, 0, 0])
    return np.diag(per_capita(y, p)) + y[:, None]*DF


@dataclass(frozen=True)
class Stability:
    classification: str
    eigenvalues: tuple[complex, ...]
    spectral_abscissa: float
    near_zero: bool


def classify_stability(y, p: Parameters, *, eigenvalue_tol=1e-8, matrix=None):
    if not math.isfinite(eigenvalue_tol) or eigenvalue_tol <= 0:
        raise ValueError('eigenvalue_tol must be finite and positive')
    eig = np.linalg.eigvals(jacobian(y, p) if matrix is None else matrix)
    largest = float(np.max(eig.real))
    label = ('unstable' if largest > eigenvalue_tol else
             'stable' if largest < -eigenvalue_tol else 'nonhyperbolic')
    return Stability(label, tuple(complex(v) for v in eig), largest,
                     bool(np.any(np.abs(eig.real) <= eigenvalue_tol)))


def seagrass_biomass(M, P, p: Parameters):
    """A.33; NaN denotes an undefined elimination, not a zero biomass."""
    loss = p.m_S + p.m_SM*M
    if p.I_0 <= 0 or p.k_S <= 0 or not 0 < loss < p.mu_S:
        return math.nan
    required = p.K_I_S*loss/(p.mu_S-loss)
    return (math.log(p.I_0/required) - p.H*(p.kw+p.k_P*P) - p.k_M*M)/p.k_S


def macroalgae_from_phyto(P, p: Parameters):
    """A.32, eliminating dissolved N from F_P=0."""
    light = p.I_0*math.exp(-p.H*(p.kw+p.k_P*P)/2)
    loss = p.m_P+p.g_P
    denominator = p.mu_P*light/(p.K_I_P+light)-loss
    if denominator <= 0:
        return math.nan
    nitrogen = p.K_N_P*loss/denominator
    return p.N_wc-p.H*(P+nitrogen)


def scalar_roots(function: Callable[[float], float], lower, upper, *,
                 grid_points=257, refinement_levels=2, residual_tol=1e-10,
                 x_tol=1e-11, dedup_tol=1e-7, grid_evaluator=None):
    """Find sign-changing roots and resolve extrema before bracketing.

    Unlike minimizing |F| around every crossing, optimizing signed extrema
    does not produce a second approximate copy of an already bracketed root.
    A finite grid still cannot guarantee detection of arbitrary root pairs.
    """
    if grid_points < 3 or refinement_levels < 0:
        raise ValueError('Require grid_points >= 3 and refinement_levels >= 0')
    if not all(math.isfinite(v) and v > 0 for v in (residual_tol,x_tol,dedup_tol)):
        raise ValueError('Root tolerances must be finite and positive')
    if not (math.isfinite(lower) and math.isfinite(upper)) or upper < lower:
        raise ValueError('Invalid root interval')
    if upper == lower:return []
    xs=np.linspace(lower,upper,(grid_points-1)*2**refinement_levels+1)
    fs=(np.array([function(float(x)) for x in xs]) if grid_evaluator is None
        else np.asarray(grid_evaluator(xs),dtype=float))
    if fs.shape!=xs.shape:raise ValueError('Grid evaluator returned wrong shape')
    extrema=[]
    left,mid,right=fs[:-2],fs[1:-1],fs[2:]
    minima=(mid<left)&(mid<right);maxima=(mid>left)&(mid>right)
    valid=np.isfinite(left)&np.isfinite(mid)&np.isfinite(right)
    for i in np.flatnonzero(valid&(minima|maxima))+1:
        sign=1 if minima[i-1] else -1
        if sign:
            opt=minimize_scalar(lambda x:sign*function(x),bounds=(xs[i-1],xs[i+1]),
                                method='bounded',options={'xatol':x_tol})
            if opt.success and math.isfinite(opt.fun):extrema.append((float(opt.x),sign*float(opt.fun)))
    nodes=sorted(list(zip(xs,fs))+extrema)
    roots=[]
    def add(x):
        x=float(x)
        if math.isfinite(function(x)) and abs(function(x))<=residual_tol:
            if not any(abs(x-r)<=max(dedup_tol,16*np.spacing(max(abs(x),abs(r)))) for r in roots):
                roots.append(x)
    for (a,fa),(b,fb) in zip(nodes[:-1],nodes[1:]):
        if not (math.isfinite(fa) and math.isfinite(fb)):continue
        if fa==0:add(a)
        if fb==0:add(b)
        if a<b and fa*fb<0:
            try:add(brentq(function,a,b,xtol=x_tol,rtol=4*np.finfo(float).eps))
            except ValueError:pass
    # Tangencies are admitted only to near machine accuracy, and only if
    # a nearby sign-changing root has not already established the crossing.
    for x,f in (nodes[0],nodes[-1]):
        if abs(f)<=2e-13:add(x)
    for x,f in extrema:
        if abs(f)<=min(residual_tol,2e-13):
            i=np.searchsorted(xs,x)
            if not any(xs[max(0,i-2)]<=r<=xs[min(len(xs)-1,i+1)] for r in roots):add(x)
    return sorted(roots)


@dataclass(frozen=True)
class Equilibrium:
    community: str
    biomass: tuple[float, float, float]
    nitrogen: float
    stability: Stability
    residual_norm: float
    per_capita_residual: float
    feasible: bool = True


def solve_equilibria(p: Parameters | None = None, *, grid_points=257,
                     refinement_levels=2, growth_tol=1e-8, ode_tol=1e-8,
                     biomass_tol=1e-8, eigenvalue_tol=1e-8):
    """Find and validate all eight classes using the five A.34 residuals.

    Zero mortality/self-shading degeneracies can generate continua instead of
    isolated roots; these are rejected explicitly by this reduced solver.
    """
    p = Parameters() if p is None else p
    for v in (growth_tol, ode_tol, biomass_tol, eigenvalue_tol):
        if not math.isfinite(v) or v <= 0:
            raise ValueError('Solver tolerances must be finite and positive')
    if p.k_S <= 0 or p.m_S <= 0 or p.m_M <= 0 or p.m_P+p.g_P <= 0:
        raise ValueError('Reduced solver requires k_S, m_S, m_M and m_P+g_P > 0')
    equilibria = []
    def accept(label, y):
        y = np.asarray(y, dtype=float)
        present = np.array([c in label for c in 'MSP'])
        if not np.all(np.isfinite(y)) or np.any(y < 0):
            return
        if np.any(y[present] <= biomass_tol) or np.any(y[~present] != 0):
            return
        N = (p.N_wc-y[0])/p.H-y[2]
        if N < -biomass_tol:
            return
        F = per_capita(y, p)
        residual = float(np.max(np.abs(y*F)))
        growth = float(np.max(np.abs(F[present]))) if np.any(present) else 0.0
        if residual > ode_tol or growth > growth_tol:
            return
        if any(np.allclose(y,e.biomass,atol=biomass_tol,rtol=1e-8) for e in equilibria):
            return
        equilibria.append(Equilibrium(label, tuple(float(v) for v in y), float(N),
            classify_stability(y, p, eigenvalue_tol=eigenvalue_tol), residual, growth))
    accept('0', (0, 0, 0))
    accept('S', (0, seagrass_biomass(0, 0, p), 0))
    # The M and P monoculture per-capita rates are strictly decreasing
    # over their feasible inventory intervals (positive losses assumed).
    # Direct bracketing also resolves roots arbitrarily close to extinction.
    def mono(i, upper):
        if upper<=0:return None
        def f(x):
            y=np.zeros(3);y[i]=x
            return per_capita(y,p)[i]
        if f(0)<=0 or f(upper)>=0:return None
        return brentq(f,0,upper,xtol=1e-12,rtol=4*np.finfo(float).eps)
    P0=mono(2,p.N_wc/p.H)
    M0=mono(0,p.N_wc)
    if M0 is not None:accept('M',(M0,0,0))
    if P0 is not None:
        accept('P',(0,0,P0))
        accept('SP',(0,seagrass_biomass(0,P0,p),P0))

    # Resident seagrass is eliminated analytically. For discovery only,
    # extend the residual continuously with S=0 outside its positive domain.
    # All reconstructed roots are checked against the ORIGINAL equations and
    # strict positive resident biomass, so the extension cannot add states.
    def search_S(M,P):
        S=seagrass_biomass(M,P,p)
        return max(S,0.) if math.isfinite(S) else 0.
    def search(label,lo,hi,reconstruct):
        if hi<=lo:return
        def residual(x):
            M,S,P=reconstruct(x)
            return per_capita((M,search_S(M,P) if 'S' in label else 0.,P),p)[0]
        def grid_residual(xs):
            # Same equations and domain extension as residual(), evaluated
            # for the entire search grid in one NumPy operation. Brent and
            # extrema correction still use the scalar residual above.
            if label=='MS':
                M=xs;P=np.zeros_like(xs)
            else:
                P=xs
                light=p.I_0*np.exp(-p.H*(p.kw+p.k_P*P)/2)
                loss=p.m_P+p.g_P
                denominator=p.mu_P*light/(p.K_I_P+light)-loss
                with np.errstate(divide='ignore',invalid='ignore'):
                    M=np.maximum(0.,p.N_wc-p.H*(P+p.K_N_P*loss/denominator))
                M=np.where(P==P0,0.,M)
            S=np.zeros_like(xs)
            if 'S' in label and p.I_0>0 and p.k_S>0:
                loss=p.m_S+p.m_SM*M
                valid=(loss>0)&(loss<p.mu_S)
                with np.errstate(divide='ignore',invalid='ignore'):
                    required=p.K_I_S*loss/(p.mu_S-loss)
                    raw=(np.log(p.I_0/required)-p.H*(p.kw+p.k_P*P)-p.k_M*M)/p.k_S
                S=np.where(valid&np.isfinite(raw),np.maximum(raw,0.),0.)
            return growth_expressions(M,S,P,p,np.exp)[0]
        for x in scalar_roots(residual,lo,hi,grid_points=grid_points,
                              refinement_levels=refinement_levels,grid_evaluator=grid_residual):
            M,S,P=reconstruct(x)
            accept(label,(M,seagrass_biomass(M,P,p) if 'S' in label else 0.,P))
    # With S present, M cannot exceed its ungrazed monoculture value.
    if M0 is not None:
        search('MS',0.,M0,lambda M:(M,0.,0.))
    if P0 is not None:
        # F_P=0 uniquely gives M(P); M decreases from M(0) to 0 at P0.
        # At P0, cancellation may leave a tiny negative M: use the exact
        # endpoint and do not introduce NaN into the last root bracket.
        def reconstruct(P):
            M=0. if P==P0 else macroalgae_from_phyto(P,p)
            if not math.isfinite(M):
                raise ArithmeticError('Ill-conditioned phytoplankton elimination')
            return max(0.,M),0.,P
        for label in ('MP','MSP'):search(label,0.,P0,reconstruct)
    return equilibria
