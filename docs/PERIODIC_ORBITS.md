# MSP periodic-orbit analysis

This replaces the earlier integration screen. No IVP integration or shooting
is used in the continuation pipeline. The separate time-series analysis remains
available for its existing purpose.

## Modules and search

`periodic_config.py` validates settings; `periodic_math.py` provides admissibility,
exclusion bounds, deterministic eigenmode guesses and Floquet classification.
`periodic_orbits.py` orchestrates discovery, reuse, deduplication, isolated workers
and exports. `periodic_native.py` solves and continues native periodic BVPs.
`periodic_worker.py` is the disposable native process entry point.

Only strictly positive MSP coexistence equilibria seed calculations, regardless
of stability. Lower-dimensional communities are excluded by the Bendixson–Dulac
result specified for this SIMP formulation; revisit that assumption if the
equations change. Existing overview/region roots and traced MSP branches are
reused. Every discovered MSP equilibrium at a selected location is investigated.

Maximum growth bounds or covering interval bounds of the reduced MSP residual
provide sufficient absence certificates. An empty finite root scan is
`msp_not_found`, not a proof. Exceptions are `equilibrium_discovery_failed`.
Successful orbit corrections are `verified_periodic`; unsuccessful attempts
never imply absence of cycles.

The coarse grid, sampled existing branches and MSP Hopf points/curves are
searched first. Nearby verified orbits provide additional guesses. Complex
Jacobian modes are prioritized; deterministic real-vector planes support
equilibria with a purely real spectrum and can also be enabled for complex
spectra with `periodic_real_modes=always`. Amplitudes are relative log-biomass
perturbations, reduced as necessary to keep initial nitrogen nonnegative.
Periods derive from modal frequencies or real-eigenvalue timescales, multiplied
by configured factors and bounded by admissible period limits.

Local axis offsets refine sites with cycles or unresolved MSP searches.
`periodic_max_locations` caps work; coverage lists scheduled locations left
unvisited and suggested further refinement. Finite guesses and finite grids
can miss disconnected branches. Completed discovery is not global completeness.

## Native continuation and stability

At a corrected Hopf, `switch_to_hopf_orbit()` generates the emerging periodic
branch and its Lyapunov coefficient classifies supercritical/subcritical
emergence. Nearly zero coefficients are unresolved. Manual and reused guesses
use `set_current_dofs()`, `activate_periodic_orbit_handler()`, and `solve()`.
Orthogonal collocation and periodic B-splines are supported.

`arclength_continuation()` traces both directions. Unstable orbits are retained,
and the continuation parameter may reverse at a fold. In 2D, both axes serve
as continuation charts with the other parameter fixed. Parameter reversals
are exported as cycle-fold brackets, not precisely localized fold-of-cycle
curves. Stability changes and turns reduce subsequent step sizes. Stops at
domain/presence limits, collapse, validation failures, native errors and limits
are recorded. Short parameter/state segments deduplicate already covered
branches; different periods or envelopes are retained.

Floquet spectra come from `orbit.get_floquet_multipliers()`. Exactly one
multiplier nearest unity is removed, and only within tolerance. Another near-unit
multiplier remains, as required at cycle folds. All three multipliers are
exported. Missing spectra or poor unity accuracy are unresolved; the other two
determine orbital stability, with a margin for discretization error. Pyoomph
computes B-spline spectra on its collocation resampling of the orbit.

## Verification

Every accepted point checks strict positive M/S/P, nonnegative dissolved
nitrogen, nontrivial amplitude and a bounded positive period on dense samples.
The algebraic inventory is `M + H*P + H*N = N_wc`; seagrass is excluded from
this inventory by the model.

The assembled BVP residual includes phase and periodicity constraints. Independent
fine-mesh native solves verify periods, extrema/means and Floquet spectra without
changing the coarse branch's arclength tangent. An additional balance defect is
computed by quadrature along the already-solved periodic curve: it evaluates
the ODE residual and does not advance a state or solve an IVP. A finite-difference
differential defect is exported diagnostically; interpolation at collocation
knots can give it larger artifacts than the integral balance defect. The balance
defect is the independent acceptance criterion. An admissible MSP equilibrium
must also exist at each accepted continuation point.

These are numerical checks, not interval proofs for every phase of an orbit.
No unstable or mesh-sensitive cycle is labeled stable. Stationary collapse,
failure and timeout never generate a stable-oscillation label.

## Isolation, settings and outputs

Each discovery attempt and each continuation arm runs in a disposable process,
with single-thread numerical libraries and no visible console windows. Native
crashes and timeouts remain local. Verified points are flushed and checkpointed
before proceeding, preserving partial branches. Cooperative stop terminates the
active worker. Every invocation uses distinct request/output files.

The continuation form exposes `periodic_*` settings: grid resolution, amplitudes,
period factors/bounds, mode/order/nodes, Newton tolerances/limits, refinement,
Floquet tolerances, arclength steps/limits, location cap and worker timeout.
Disable `periodic_analysis` for equilibrium-only runtime. Old configurations
that explicitly used `oscillation_checks=off` retain the disabled preference;
old integration-day settings are not used by periodic analysis.

`periodic_H*.json` contains branches in continuation order, arms, Hopf coefficients,
location statuses, settings, attempts and coverage. `periodic_branches_H*.csv`
contains each verified point, physical/normalized parameters, period, areal
biomass extrema/means, full Floquet spectra and stability. Worker JSONL records
retain reusable log-state samples; request JSON and native logs preserve inputs
and failure details. Outputs are checkpoint artifacts, not automatic resume.

In 1D, filled envelopes join only adjacent stable points on one continued arm.
Unstable boundaries are dashed and unresolved stability is dotted. Folds retain
continuation order. In 2D, symbols mark verified locations/transects without
extrapolating cycle presence across equilibrium polygons. `M + ○(MSP)` requires
an independently verified stable M equilibrium and stable MSP orbit at the
same parameters. The 2D biomass maps continue to depict equilibria only.

## API references

- [Hopf switching](https://pyoomph.readthedocs.io/en/stable/tutorial/temporal/orbit/hopf_switch.html)
- [Manual orbit initialization](https://pyoomph.readthedocs.io/en/latest/tutorial/temporal/orbit/manual_orbit.html)
- [API source](https://pyoomph.readthedocs.io/en/latest/_modules/pyoomph/generic/problem.html)

The installed pyoomph build is the execution authority. Unsupported APIs fail
explicitly; there is no fallback to time integration.
