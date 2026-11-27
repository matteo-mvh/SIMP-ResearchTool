# Add an analysis without modifying the GUI

Read `scripts/example_documented/{analysis.toml,run.py}`; both files contain extensive comments.

## 1. Duplicate the folder

Copy `scripts/example_documented/` to `scripts/my_simp_analysis/`.
Give the manifest a unique `id`, e.g. `simp_equilibria`, plus its visible name, group, and description.

## 2. Describe inputs in `analysis.toml`

Use `[parameters.OPTION_NAME]` for each input. Valid types: `float`, `integer`, `choice`, `boolean`, `string`, `seed`.
The GUI generates the input fields and separates them by the supplied `category`.
Set a `default`, optional `min`, `max`, `unit`, and `description`.

**All declared inputs are editable.** If there are many scientific parameters, put them
in a separate TOML/JSON file and set `parameter_file = "model_parameters.toml"` in
`analysis.toml`. The parameter file uses the same `[parameters.NAME]` format, and
all its entries automatically appear in the GUI. Manifest entries override
same-named file entries. See `scripts/example_documented/model_parameters.toml`.
This is declarative discovery: Python constants embedded inside arbitrary scripts
are NOT automatically parsed or edited. Declare run-time parameters in the manifest
or parameter file, and read them from `p` inside `run.py`.

## 3. Implement `run.py`

Call `p, ctx, job = load_args()`. Your code must accept `--config` and `--output` through this helper. Import your real model functions from `model/`; do not put them in GUI code.

- `p['kw']`, `p['depth']`, etc.: values selected in the GUI and validated.
- `p`: the **effective parameter set**. Starting with the chosen seed, declared fields and individually edited seed-only fields override its original values. All fields are editable in the Create Job form.
- `job['seed_values']`: immutable original saved parameter values before editing.
- `job['seed_overrides']`: changes compared with the selected seed, for audit and reproduction.
- `job['seed_source_job_id']`: original analysis job that created that seed.
- `ctx.folder`: run's unique output directory.
- `ctx.checkpoint(percent, phase)`: update GUI progress and honor pause/stop.
- Save plots to `ctx.folder / 'figures' / 'name.png'` and data in `ctx.folder`.

Use `ctx.checkpoint()` at safe intervals, not inside an expensive tight loop. An uninterrupted SciPy/PyOOMPH solve cannot be paused until control returns to your script. No parallel jobs are launched by the scheduler.

Return `0` on success, `130` on a cooperative stop; raise errors to get a failed job with an error log.

## 4. How sensitivity seeds are created and selected

**Any completed job automatically becomes a reusable run seed (`RUN-...`); any analysis may additionally emit thousands of generated Seed-IDs using `seeds.csv`. Do **not** submit a new job for every parameter draw merely to save it. Instead write a streaming CSV named `seeds.csv` in its results directory with columns:

```csv
seed_id,kw,g_M,m_SM,metric__bistability,metric__delta_S
S-abcd1234-000001,0.25,0.05,0.00003,1,91.5
S-abcd1234-000002,0.48,0.01,0.00007,0,4.0
```

The worker indexes this file into SQLite after a successful run. Regular columns (e.g. `kw`) are parameter values; prefixed `metric__` columns are *read-only output metrics*, shown when browsing but never silently used as model inputs. Seed IDs must be unique across runs; include the run ID (`job['id'][:8]`) and an incrementing index.

Use **Seed library** to filter by any source job, seed type, ID, or bistability classification, and page through only 80 matches at a time. Select an ID in **Create job → Find seed…** or directly from the library. The complete original seed is copied into the resulting job config. Parameters already defined in the receiving analysis populate its controls, while additional inherited values appear under “Additional seed parameters” with editable fields. Edit those freely; the final effective parameters and deviations from the original seed are both saved. Verify which parameters are supported by a downstream analysis before running it.

**Random seed** is different: this is the integer that initializes the sampling generator and makes the same sensitivity sweep reproducible. **Seed-ID** identifies a single saved parameter vector from that sweep.

## 5. Test your analysis

1. Refresh the GUI. Your analysis should appear under its manifest `group`.
2. Create a 5-point named draft and queue it; inspect `Results` and `stdout.log`.
3. Confirm plots and data were saved and the output values respond to changed parameters.
4. Run the same `run.py --config ... --output ...` from a terminal and compare results.

No `app/ui.py` modifications are required.
