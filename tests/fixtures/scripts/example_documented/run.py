"""EXAMPLE ANALYSIS: read this file when writing a new SIMP analysis.

PURPOSE: show the precise contract between your analysis code and the GUI.
This is NOT SIMP and should never be used as a scientific result.

WHAT TO COPY:
  1. Duplicate this FOLDER and rename it.
  2. Change the id/name/parameters in analysis.toml.
  3. Replace the synthetic numerical block below with your model function.
  4. Keep the load_args(), checkpoint(), outputs and __main__ entrypoint.
  5. Click Refresh in the GUI: no GUI code changes are needed.

The worker runs (from the project root):
    python scripts/example_documented/run.py --config RUN/config.json --output RUN

config.json contains: job name, all final validated input parameters, and,
if selected, the ORIGINAL seed_values copied from any prior run or sample.

All input parameters shown in the GUI are described by analysis.toml and
model_parameters.toml. All are editable, and selecting a seed just PREFILLS them.
Additional seed-only parameters are also editable in the form.

The worker is sequential (one job at a time). The GUI never executes this file.
"""
from __future__ import annotations
import csv
import math
import sys
from app.core.control import load_args, StopRequested
from app.core.demo_helpers import plt_module, finish_plot


def main():
    # Returns p = validated GUI parameters; ctx = run-local output controls;
    # job = entire run config (including full selected seed metadata).
    p, ctx, job = load_args()
    print(f"Starting tutorial run: {job['name']}", flush=True)
    print(f"Saved Seed-ID: {job.get('seed_id') or '(none)'}", flush=True)
    print(f"Effective parameters: {p}", flush=True)
    print(f"Seed overrides: {job.get('seed_overrides', {})}", flush=True)
    if job.get('seed_values'):
        print(f"Complete source seed: {job['seed_values']}", flush=True)

    x_values, y_values = [], []
    try:
        # Replace this loop with model, continuation, sensitivity, etc.
        # Always checkpoint after small blocks, not every CPU-intensive step.
        for i in range(p['points']):
            x=i/(p['points']-1)
            # The tutorial also uses coefficients defined ONLY in model_parameters.toml.
            # This remains a synthetic function, not ecological modelling.
            synthetic_scale=1 + 0.4*(p['mu_M']-0.174) + 0.1*(p['mu_P']-1.0)
            synthetic_decay=p['kw']*p['depth'] + 0.3*p['mu_S']
            y=p['amplitude'] * synthetic_scale * math.exp(-synthetic_decay*x) * (0.8+0.2*math.cos(6*x))
            x_values.append(x)
            y_values.append(y)
            if i % max(1, p['points']//100)==0 or i==p['points']-1:
                ctx.checkpoint(90*(i+1)/p['points'], f'Computed {i+1}/{p["points"]}')

        # Always save machine-readable outputs to THIS job's own folder.
        with (ctx.folder/'trajectory.csv').open('w',newline='',encoding='utf-8') as fp:
            writer=csv.writer(fp)
            writer.writerow(['relative_time','synthetic_signal'])
            writer.writerows(zip(x_values,y_values))
        ctx.checkpoint(94, 'Saving figures')

        # Headless plotting: no pop-up windows and no GUI compute overhead.
        plt=plt_module()
        fig,ax=plt.subplots(figsize=(8,4))
        ax.plot(x_values,y_values,color='#16803F',lw=2)
        ax.set(xlabel='Relative time',ylabel='Synthetic response',title='Tutorial — not SIMP data')
        fig.tight_layout()
        finish_plot(fig,ctx,'tutorial_response.png')
        ctx.progress(100,'Complete',True)
        return 0
    except StopRequested:
        # Cooperative stop preserves previous output. The worker sees exit 130.
        print('Stopped by user',flush=True)
        return 130


if __name__=='__main__':
    raise SystemExit(main())
