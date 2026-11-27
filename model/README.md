# Shared SIMP model

`shared_eqn.py` implements thesis equations 3.8-3.14 and Appendix A.3 (A.28-A.41, Algorithm 1). Biological defaults follow Table 3.4; reference environmental values follow section 3.4: N_wc=20, H=4, I_0=384, kw=0.30.

State order is **(M, S, P)**. M and S are mmol N/m², P and dissolved N are mmol N/m³, N_wc is mmol N/m², H is metres, and time is days.

## Equilibria and stability

```python
from model.shared_eqn import Parameters, solve_equilibria

p = Parameters(N_wc=200, H=4, kw=0.3)
for eq in solve_equilibria(p):
    print(eq.community, eq.biomass, eq.stability.classification)
    print(eq.stability.eigenvalues, eq.residual_norm)
```

Five scalar residuals reconstruct all eight classes: 0, M, S, P, MS, MP, SP, MSP. Multiple equilibria of one class may be returned. Undefined eliminations are excluded, present species must exceed `biomass_tol`, and candidates must pass per-capita and full ODE residual checks. Dissolved nitrogen is `(N_wc-M)/H-P`; seagrass is excluded from this inventory.

The full analytical Jacobian retains invasion eigenvalues for absent species. Classification is `stable`, `unstable`, or `nonhyperbolic` using `eigenvalue_tol` (default 1e-8 per day). `near_zero` also flags eigenvalues close to the imaginary axis when another eigenvalue already proves instability. `classify_stability` assumes an equilibrium; `solve_equilibria` checks stationarity first.

Grazing choices `(gamma, beta)` are `(0,1)` linear, `(1,1)` saturating, and `(1,2)` sigmoidal. Other combinations raise `ValueError`.

Root search uses nested grids, Brent brackets and local minimization of absolute residuals to find tangencies. Increase `grid_points` and `refinement_levels` to check resolution. Finite scans cannot guarantee completeness, especially near folds; check selected configurations by multistart or continuation. Numerical tolerances are implementation choices, not values prescribed by the appendix. Roots within `biomass_tol` of a class boundary are excluded from that positive class.

The reduced solver requires positive `k_S`, `m_S`, `m_M`, and `m_P+g_P`. Zero values can invalidate eliminations or create non-isolated equilibria and raise an explicit error. The ODE supports nonnegative coefficients, including zero interaction strengths.

## Time series using the same equations

```python
import numpy as np
from scipy.integrate import solve_ivp
from model.shared_eqn import Parameters, rhs, jacobian

p = Parameters()
solution = solve_ivp(
    rhs, (0, 365), (2, 30, 1), args=(p,),
    t_eval=np.linspace(0, 365, 366), rtol=1e-9, atol=1e-11,
)
if not solution.success:
    raise RuntimeError(solution.message)
# solution.y rows are M, S, P.
# For an implicit method such as Radau, also pass:
# jac=lambda t, y, p: jacobian(y, p)
```

Initial biomasses must be nonnegative and satisfy M+H*P <= N_wc. Equations are not clipped or projected; scripts must check integration success and physical-domain errors. An initially absent species stays absent; there is no recruitment term.

## Parameters and seeds

`Parameters` is the shared biological/environmental parameter definition. Use it directly or call `Parameters.from_mapping(effective_values, allow_extra=True)` when a GUI parameter dictionary also contains run controls. Default strict mode rejects unknown names. Canonical names include `H`, `I_0`, `N_wc`, and `kw`; demo names such as `depth` are not aliases.

Declare editable inputs in `analysis.toml` or its linked parameter file using these exact names. Continuation manifests opt into the shared schema with `form = "continuation_1d"` or `"continuation_2d"`. The GUI does not import the pyoomph runner or run scripts; the worker passes effective values after seed overrides. Existing demo scripts remain synthetic.

`growth_expressions` and `resource_expressions` provide the same equation algebra to NumPy and pyoomph. Native stationary correction and pseudo-arclength continuation live in `pyoomph_continuation.py`; region checks live in `continuation_regions.py`. The shared analytical Jacobian supplies the full stability classification, including invasion eigenvalues. See the main README for task settings, exports, validation and limitations.
