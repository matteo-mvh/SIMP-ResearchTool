# Working, fast equilibrium baseline — 2026-11-27

The user reported that the changes made on November 26 made equilibrium
discovery substantially faster and more stable, finding the stable equilibria
in their configurations. Treat this as the accepted working baseline; do not
replace the reduced solver or continuation architecture without regression
evidence. This is a user assessment, not a mathematical completeness claim.

The exact source before the oscillation additions is preserved in
`baselines/2026-11-27-working-fast/source.zip`, with SHA-256 hashes in
`manifest.json` alongside it. There was no Git repository in this workspace.

Structure:

- `model/shared_eqn.py`: common equations, analytical Jacobian, reduced roots.
- `model/pyoomph_continuation.py`: discovery and native branch/event tracing;
  vectorized root matching and cached vectorized branch-distance calculations.
- `model/continuation_regions.py` and `continuation_refinement.py`: independent
  region checks and bounded conflict-driven refinement with cached equilibria.
- `scripts/continuation_runner.py`: depth orchestration, recovery exports,
  optional independent processes, and figures through `continuation_plots.py`.
- `model/periodic_*.py`: separate MSP periodic BVP workflow; see
  `docs/PERIODIC_ORBITS.md`. This supersedes the earlier integration screen.

The oscillation feature must preserve all equilibrium outputs and labels.
Hopf events are hints, not proof of stable cycles. Periodic BVP correction,
mesh refinement and Floquet spectra provide numerical verification. The earlier
integration screen has been removed at the user's request.
Do not fill an entire 2D equilibrium polygon based only on local cycle evidence.

Performance check during this change: 30,000 RHS calls took 3.77 s before and
3.11 s with duplicate state validation removed, under concurrent test load.
This is a callback microbenchmark, not a whole-job speedup measurement. The
reduced equilibrium algorithm was left unchanged. Cycle screening adds work
and can be disabled with `periodic_analysis=off`.

Historical verification of the earlier integration screen: the 60-test suite passed after updating two stale
mock fixtures. Both original failures were reproduced against the archived
baseline before fixture changes. Two further trigger/coverage tests and focused
model/time-series tests passed. PNG/PDF layouts were rendered and inspected.
With the reference parameters and H=4, checks at N_wc=80 and 82 detected MSP
cycles with periods about 456 and 455 days. N_wc=65, 78 and 100 remained
unresolved within 20,000 days; these results are not evidence against cycles.
