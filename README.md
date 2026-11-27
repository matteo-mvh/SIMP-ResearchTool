# SIMP Control Center

A modular Tkinter desktop app with a separate sequential worker, reusable seed library, and shared SIMP model.

## Folder layout

- `app/`: GUI (`ui.py`), job/discovery/storage backend (`core/`), branding (`assets/`), and reserved presets.
- `model/`: shared scientific code. `shared_eqn.py` contains the SIMP ODE, analytical Jacobian, reduced equilibrium solver, stability classification, and parameter defaults.
- `scripts/`: independently discoverable experiment folders, each with `analysis.toml` and `run.py`.
- `results/`: run configurations, figures, data, and logs.
- `state/`: existing job/seed SQLite database and worker heartbeat. Kept at its original location to preserve history.
- `tests/` and `docs/`: verification and developer instructions.

`start_app.py`, `start_worker.py`, and the two Windows launchers remain at the root for convenient launching. **1D continuation · SIMP** and **2D continuation · SIMP** use the real model and native pyoomph continuation. **Time series · SIMP** integrates the shared equations. Synthetic examples are retained only under `tests/fixtures/scripts`, outside app discovery.

## Install and launch

Use Python 3.11+ with Tkinter:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe start_app.py
```

Double-click `Start_SIMPCenter.bat` for normal use. It prefers this project's virtual environment. `Start_Worker_Console.bat` does likewise. The GUI and worker use the same interpreter; numerical dependencies are NumPy and SciPy, and plotting dependencies are Matplotlib and Pillow.

## Add or remove scripts

Read [the analysis guide](docs/ADDING_AN_ANALYSIS.md) and [the documented example](tests/fixtures/scripts/example_documented/run.py).

Each direct subfolder of `scripts/` with a valid `analysis.toml` and Python entrypoint appears automatically on **Create job → Refresh**. The manifest defines the visible title, group, description, and editable inputs. Script header comments document how to run it; the GUI reads TOML metadata without importing the script. Add or remove experiment folders without changing GUI code.

The optional `parameter_file` links a TOML/JSON parameter schema within the experiment folder; manifest entries override that schema. Scripts receive final values through `app.core.control.load_args()`. Shared scientific functions are imported from `model.shared_eqn`. See [model usage and time-series example](model/README.md).

## Jobs, results, and seeds

**Parameter inspection**, below Seed library, provides a live split view of all
24 biological and environmental model parameters and their equilibria. Edits
recalculate after a 300 ms typing pause in a background thread; outdated
calculations are discarded. Reset defaults restores the thesis reference values.
Stable, unstable, and near-zero/unresolved equilibria are listed separately,
with resident biomasses, dissolved nitrogen, absent-species invasion rates,
and eigenvalues. Community names label the points. All displayed biomasses and
nitrogen use mmol N/m² (P and N concentrations are multiplied by depth).
The plot compares the largest eigenvalue real part with zero.
Values survive navigation during the current app session; inspection does not
change saved jobs, seeds, or global defaults. Invalid input and unsupported
degenerate reductions display an error rather than stale results.
Use **Browse seed library** or enter a Seed-ID at the top to populate the fields
and recalculate. Missing parameters use thesis defaults rather than values left
over from a previous seed. Legacy seed names `depth` and `nwc` map to `H` and
`N_wc`; canonical names take precedence. Non-model fields are listed as ignored.

The worker runs one computation at a time and survives closing the GUI. Scripts accept `--config` and `--output`; output goes to `results/runs/<run>/`. Call `ctx.checkpoint()` to support cooperative pause/stop and progress.

Every completed job becomes a reusable `RUN-...` seed. Any script can also emit `seeds.csv` with `seed_id`, parameter columns, and optional `metric__...` columns. The library indexes these in SQLite and pages large collections. Selecting a seed prefills editable fields and retains the source values, final effective parameters, overrides, and provenance in `config.json`.

Set `SIMP_DATA_DIR` to store `state/` and `results/` elsewhere. A local nonsynced directory is preferable for the SQLite WAL database. Existing state and results were preserved during the folder reorganization.

## Native continuation jobs

Choose a SIMP continuation job in Create job. The split editor shows biological
parameters on the left and continuation settings on the right. Load a saved seed,
choose one or two continuous parameters, their ranges and resolutions, and select
1, 2, or 4 depth panels. Selected axes and H are disabled on the left; their values
come from the task settings. Gamma/beta remain discrete grazing choices.

The scalar solver and analytical Jacobian discover positive resident starts and
classify full-system stability, including absent-species invasion rates. Pyoomph
performs stationary correction, pseudo-arclength branch continuation, fold/Hopf
tracking and augmented invasion-curve continuation. Both use the same growth
expressions in `model/shared_eqn.py`. TCC is selected explicitly on Windows, so
Microsoft C++ Build Tools are not required.

Resolution sets the maximum normalized arclength step (1/resolution); adaptive
stepping can produce more points. Discovery samples and 2D transects control
initial branch/event coverage separately. Increase these near narrow coexistence
windows or folds. Depth is always a scenario, not a continuation parameter.

The previous plot layouts and palettes are retained: four-component 1D overview,
seagrass branches with a zero-seagrass strip, and depth-panel 2D state/biomass maps
with side legends. Depth comparisons use one panel, two side by side, or a 2×2 grid.
Every figure is saved in `figures/` as a 270-DPI PNG and a PDF with embedded fonts.
P and N are displayed as areal values; CSV files retain concentrations as well as
`P_area` and `N_area` for reproducibility.

Outputs include branch/curve CSVs, overview and event CSVs, region samples,
diagnostics JSON, and `summary.json`. In 2D, native curves divide the parameter
domain; independent scalar equilibrium checks classify the regions. Conflicting
checks or incomplete curve endpoints mark affected regions `?`. Biomass maps
interpolate checked samples within each region and select the stable equilibrium
with maximum seagrass when several coexist. They are not additional continuation
solutions. Native curve endpoints inside the domain are marked with crosses.

Review diagnostics for failed candidate corrections, step limits, and uncovered
starts. Finite discovery cannot prove complete coverage. Hopf curves identify
linear bifurcations. MSP periodic-orbit analysis now follows continuation
(see below). These are research computations,
not a claim that every parameter choice is globally resolved.

### MSP periodic orbits (no time integration)

The accepted fast equilibrium source is archived and described in
[the working baseline](docs/WORKING_BASELINE.md). A separate periodic BVP stage
uses pyoomph orthogonal collocation or periodic B-splines, never time integration.
Only parameter combinations with a verified positive MSP equilibrium are
investigated, including stable equilibria. Lower-dimensional communities are
analytically excluded for this formulation. Existing equilibria and Hopf events
are reused; empty numerical searches are not treated as certificates of absence.

Hopf switching computes the first Lyapunov coefficient. Independently, periodic
guesses around MSP equilibria use Jacobian modes, several amplitudes and periods,
including real-mode planes when no complex pair exists. Discovered orbits are
continued in both arclength directions, through unstable segments and parameter
turning points. Floquet classification removes exactly one trivial multiplier;
near-unit or mesh-sensitive results remain nonhyperbolic or unresolved.

In 1D, stable cycle envelopes are filled in the existing colors; unstable
envelopes are dashed. Continuation order preserves folds. In 2D, verified stable
cycle locations use circles, including `M + ○(MSP)` for coexistence with a stable
M equilibrium. Unstable/unresolved cycles use open triangles. Equilibrium maps
remain intact; no entire polygon is assigned a cycle from local evidence.

`periodic_H*.json` and `periodic_branches_H*.csv` contain periods, biomass
minima/maxima/means, Floquet spectra, stability, mesh/physical checks, Hopf
classifications, search outcomes and continuation stop reasons. Worker folders
retain requests, native logs and reusable orbit samples. Each native attempt
runs in its own process with a timeout; verified points are checkpointed before
the next step. Coarse discovery and targeted refinement have explicit limits,
and unvisited locations are reported. Configure the `periodic_*` controls or
turn `periodic_analysis` off for equilibrium-only runs. See
[periodic analysis details](docs/PERIODIC_ORBITS.md) for validation and limitations.

The 2D state legend lists only regions visible at the figure's display scale;
subpixel slivers remain in the data. `M+P` means alternative stable M and P
equilibria, whereas `MP` is one coexistence equilibrium. Grey means unresolved,
not monostable or absent. Large-inventory phytoplankton roots use the direct
growth residual to avoid inverse-elimination conditioning errors. Native boundary
corrections can refresh the resident equilibrium guess before retrying pyoomph.

2D jobs now refine inconsistent regions automatically. Discovery combines uniform
and edge-clustered parameter samples with a denser root grid. The reduced solver
uses the feasible P interval for MP/MSP branches so narrow branches receive useful
resolution even at large nitrogen inventories. Conflicting region checks seed
axis-aligned local transects; changes in root counts/stability are bisected before
native fold/Hopf/invasion correction and tracing. Newly found curves rebuild the
regions, with old conflict witnesses retained unless they lie within the numerical
boundary tolerance of a newly resolved curve. Equilibrium checks are cached.

The 2D form exposes refinement rounds (default 3) and local searches per round
(default 12). Each local search is bounded to 25 equilibrium checks and 24 native
correction attempts; at most four dangling ends are retried per round. Duplicate
curves/attempts are skipped, and two rounds without progress stop refinement.
`refinement_H*.json` records searches, added curves, unresolved area and stop reason;
region JSON records whether conflicts, marginal equilibria or incomplete boundaries
remain. Initial native curves and each completed refinement round are checkpointed
to disk, so interruption does not discard all previously traced curves. These
are recovery artifacts, not an automatic resume feature. These safeguards improve
recovery, but cannot certify global completeness.

## Verification

In Results, select a finished/stopped/failed run and choose **Delete result**.
After confirmation this removes its output folder and generated seeds; other
jobs keep their copied seed parameter snapshots. Active jobs must be stopped first.

Windows workers and native compiler subprocesses run without command windows.
Continuation progress reports discovery and branch tracing within each depth.
After upgrading, restart the app so its worker health checks use the updated code.
The status strip always gives a provisional ETA, then refines it using live
stage progress and historical calculation and plotting times separately.
Timing history survives result deletion; newly measured stage times exclude pauses.
Estimates scale with depth count, computational settings and requested plots.
Branch complexity and convergence can still change runtime substantially.

Time series uses the shared SIMP RHS and an adaptive BDF integrator. Choose
duration (days), saved sample count, areal initial M/S/P, and 1/2/4 depths.
Plot groups `MS, S, P` produce three figures, with depth panels in each.
`N` can also be included. Every group exports PNG and PDF; per-depth CSVs
contain time and M/S/P/N in mmol N/m². P is converted to volumetric units
only inside the solver. Initial areal biomasses are identical across depths.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover discovery, worker execution, results and seed round trips, analytical Jacobian finite differences, monoculture solutions, grazing variants, tangencies/invalid domains, multistart comparison, and time integration.

The equilibrium scan cannot guarantee finding every root near unresolved folds. Increase resolution and independently verify selected research configurations. The reduced solver explicitly rejects degeneracies that invalidate its elimination formulas; see the model guide.

## LHS sensitivity

The sensitivity form offers an Include/Exclude button per parameter. Included
parameters have minimum/maximum fields; excluded parameters retain a fixed value.
Defaults come from `model/shared_eqn.py`. Initial 50–150% ranges are suggestions,
not thesis uncertainty bounds. The right panel contains sample count, random
seed, CPU workers and root-search resolution. Sampling is independent uniform
randomized Latin hypercube sampling (integer grazing switches are discrete).
Only the shared reduced equilibrium solver and analytical Jacobian classify states;
neither pyoomph nor time integration is used.

Bistability means at least two distinct strictly stable equilibria; this also
includes multistability. ΔS is max(S) − min(S) across those states, in mmol N/m².
It is blank when bistability is absent or unresolved, and can validly be zero
when stable states have equal S. Marginal/error outcomes remain flagged and
are excluded from the probability denominator. Finite scans can miss roots;
this is the probability of detected equilibrium bistability under the selected
input distribution, not the probability of reaching a particular state.

Every parameter set is saved in `seeds.csv` with bistability/ΔS metrics for the
seed library. `stable_states.jsonl` records stable states in areal units and their
largest real eigenvalue. `lhs.npy` and `metrics.npy` retain disk-backed numerical
arrays (metric columns: bistability, ΔS, stable count, marginal count).
CSV writing streams results; only bounded batches are dispatched to hidden
parallel processes. LHS generation requires O(sample count) temporary memory.

Figures include full-sample probability/convergence, red probability strips,
ΔS strips, bin support, per-parameter conditional probabilities, ΔS scatter and
input distributions, stable-count/ΔS histograms and CDF, input correlations,
and pairwise maps for up to six highly ranked parameters. All export PNG/PDF.
Conditional bins use all samples; other diagnostics and predictive models use
a reproducible random subset capped at 20,000. `conditional_bins.csv` contains
exact bin endpoints, counts and rates. Empty bins remain blank.

Sensitivity rankings use random forests and held-out permutation importance
for bistability (increase in Brier error) and ΔS (increase in squared error).
Validation scores and constant baselines are saved; weak models and insufficient
outcome variation are identified explicitly. These are predictive associations,
not causal effects or Sobol indices. Methods follow the
[LHS stratification definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html)
and [held-out permutation importance guidance](https://scikit-learn.org/stable/modules/permutation_importance.html).
