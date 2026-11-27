"""Estimate basin volume fractions at one fixed SIMP parameter set.

Uniform Monte Carlo on the feasible M, S, HP prism; numerical classification
uses the same attractor verification and convergence checks as phase_space.
No interpolation of basin membership and no change to continuation.
"""
from __future__ import annotations
import csv
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import numpy as np
from app.core.control import load_args, StopRequested
from model.shared_eqn import Parameters, solve_equilibria, seagrass_biomass
from scripts.phase_space import make_targets, orbit_samples, init_worker, classify_seed

from scripts.basin_fast import BasinPool
from scripts.basin_adaptive import adaptive_samples, weighted_fractions, to_state

UNRESOLVED = '#a6a6a6'


def sample_initial(p, upper_s, n, seed):
    """Reflect a uniform square into the nitrogen triangle (constant density)."""
    r = np.random.default_rng(seed).random((n, 3))
    mp = r[:, :2].copy()
    reflected = mp.sum(axis=1) > 1.
    mp[reflected] = 1.-mp[reflected]
    # The continuous distribution assigns zero probability to faces. Resample
    # exact floating-point boundary points rather than clip and bias the measure.
    invalid = (mp.min(axis=1) <= 0.) | (mp.sum(axis=1) >= 1.) | (r[:,2] <= 0.)
    if invalid.any():
        raise ValueError('Random sample on a domain face; choose another random seed')
    return np.column_stack((p.N_wc*mp[:,0], upper_s*r[:,2], p.N_wc*mp[:,1]/p.H))


def wilson(k, n):
    """95% marginal binomial interval for observed classification frequency."""
    z = 1.959963984540054
    q = k/n
    d = 1+z*z/n
    mid = (q+z*z/(2*n))/d
    rad = z*math.sqrt(q*(1-q)/n+z*z/(4*n*n))/d
    return max(0., mid-rad), min(1., mid+rad)


def fractions(labels, targets, volume):
    counts = Counter(int(i) for i in labels)
    n = len(labels)
    unresolved = counts[-1]
    rows = []
    for i, target in enumerate(targets):
        if target['stable'] != 'stable':
            continue
        k = counts[i]
        lo, hi = wilson(k, n)
        rows.append(dict(index=i, attractor=target['id'], kind=target['kind'], color=target['color'],
                         count=k, fraction=k/n, observed_ci_low=lo, observed_ci_high=hi,
                         possible_fraction_upper=(k+unresolved)/n, sampled_volume=volume*k/n))
    lo, hi = wilson(unresolved, n)
    rows.append(dict(index=-1, attractor='Unresolved', kind='unresolved', color=UNRESOLVED,
                     count=unresolved, fraction=unresolved/n, observed_ci_low=lo, observed_ci_high=hi,
                     possible_fraction_upper=unresolved/n, sampled_volume=volume*unresolved/n))
    return rows


def discover_targets(p, eq, enabled, checkpoint):
    """Reuse existing verified cycle search; never infer a cycle from extrema."""
    from scripts.continuation_runner import analytical_cycle_exclusion, search_location
    exclusion = analytical_cycle_exclusion(p)
    attempts = []
    orbit, period = None, None
    status = 'disabled'
    if exclusion:
        status = 'analytically_excluded_MSP'
    elif enabled:
        status = 'not_confirmed'
        try:
            found = search_location(p, 0., eq, None, checkpoint, attempts.append)
            if found:
                orbit = orbit_samples(p, found['seed'])
                if orbit is not None:
                    period = float(found['seed']['period'])
                    status = 'verified'
                else:
                    status = 'geometry_unavailable'
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
            status = 'search_failed'
            attempts.append(dict(reason=str(exc)))
    return make_targets(p, eq, orbit, period), dict(status=status, exclusion=exclusion, attempts=attempts)


def evaluate(item):
    """Avoid storing dense trajectories for a volume-only estimate."""
    index, state = item
    row = classify_seed((tuple(state), 0, 0))
    row.pop('path', None)
    row.pop('s_layer', None)
    row.pop('p_layer', None)
    return index, row


def calculate(p, targets, states, days, workers, checkpoint, store):
    n = len(states)
    start = time.monotonic()
    if workers == 1:
        init_worker(p, targets, days)
        for i, state in enumerate(states):
            checkpoint(i, n, time.monotonic()-start)
            _, row = evaluate((i, state))
            store(i, row)
        return
    # Keep a bounded queue so pause/stop does not launch the entire sample.
    pool = ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(p, targets, days))
    pending = set()
    next_index = done = 0
    try:
        while done < n:
            checkpoint(done, n, time.monotonic()-start)
            while next_index < n and len(pending) < workers:
                pending.add(pool.submit(evaluate, (next_index, states[next_index])))
                next_index += 1
            finished, pending = wait(pending, timeout=.5, return_when=FIRST_COMPLETED)
            for future in finished:
                i, row = future.result()
                store(i, row)
                done += 1
    finally:
        for future in pending:
            future.cancel()
        # Running integrations finish their current trajectory on graceful stop.
        pool.shutdown(wait=True, cancel_futures=True)


def export_rows(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plots(folder, p, upper_s, states, labels, targets, rows, adaptive=False):
    from scripts.continuation_plots import plt, save
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    colors = {r['index']:r['color'] for r in rows}
    names = {r['index']:r['attractor'] for r in rows}
    folder.mkdir(exist_ok=True)
    # Include zero-frequency stable targets as well as unresolved mass.
    fig, ax = plt.subplots(figsize=(8.3, max(3.2,.48*len(rows)+1.8)), layout='constrained')
    yy = np.arange(len(rows))
    q = np.array([r['fraction'] for r in rows])
    lo = np.array([r['observed_ci_low'] for r in rows])
    hi = np.array([r['observed_ci_high'] for r in rows])
    ax.barh(yy, q, color=[r['color'] for r in rows], alpha=.85)
    ax.errorbar(q, yy, xerr=np.maximum(0., np.vstack((q-lo,hi-q))), fmt='none', ecolor='.25', capsize=3)
    for y, r in zip(yy, rows):
        ax.text(min(.99, r['fraction']+.02), y, f"{r['fraction']:.1%} ({r['count']})", va='center', ha='right' if r['fraction']>.8 else 'left')
    ax.set(yticks=yy, yticklabels=[r['attractor'] for r in rows], xlim=(0,1.04), xlabel='Fraction of uniformly sampled initial-condition volume',
           title=f'Basin fractions · H={p.H:g} m · N_wc={p.N_wc:g} · n={len(labels)}')
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    fig.supxlabel(('95% marginal Hoeffding bounds (weighted audit)' if adaptive else '95% marginal Wilson intervals') + '; unresolved mass is not redistributed.', fontsize=8)
    save(fig, folder, 'basin_volume_fractions')
    # A sampled cloud, not a reconstructed separatrix. All axes are areal.
    fig = plt.figure(figsize=(9,7)); ax=fig.add_subplot(111, projection='3d')
    for i in colors:
        points=states[labels==i]
        if len(points): ax.scatter(*points.T, s=13, alpha=.55, color=colors[i], depthshade=False, label=names[i])
    for t in targets:
        if t['kind']=='cycle': ax.plot(*np.asarray(t['path']).T, color=t['color'], lw=2.5)
        else: ax.scatter(*t['state'], marker='*' if t['stable']=='stable' else 'x', color=t['color'], s=85, depthshade=False)
    ax.set(xlabel='M₀ / M', ylabel='S₀ / S', zlabel='HP₀ / HP', title='Sampled basin membership and attractors (mmol N/m²)')
    ax.legend(loc='upper left', fontsize=8); fig.tight_layout(); save(fig,folder,'basin_3d')
    # Slab projections are clearly labelled; not exact fixed-P slices.
    fig, axes=plt.subplots(1,3,figsize=(12,4.5), sharex=True, sharey=True, layout='constrained')
    for j, ax in enumerate(axes):
        low, high=p.N_wc*j/3,p.N_wc*(j+1)/3
        mask=(states[:,2]>=low)&(states[:,2]<high)
        for i,c in colors.items():
            q=states[mask&(labels==i)]
            if len(q): ax.scatter(q[:,0],q[:,1],s=19,color=c,alpha=.8,edgecolors='none')
        ax.set(xlim=(0,p.N_wc),ylim=(0,upper_s),xlabel='M₀ [mmol N/m²]',title=f'HP₀: {low:.2g}–{high:.2g}\nn={mask.sum()}')
        ax.grid(alpha=.15)
    axes[0].set_ylabel('S₀ [mmol N/m²]')
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color=c,label=names[i]) for i,c in colors.items()], loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle('Projected samples in three initial-phytoplankton slabs (not exact slices)')
    save(fig,folder,'basin_projected_slabs')


def run(values, ctx):
    started = time.monotonic()
    p = Parameters.from_mapping(values, allow_extra=True)
    if p.N_wc <= 0:
        raise ValueError('Positive N_wc required for a nonzero 3D sampling volume')
    n=int(values.get('basin_samples',256)); days=float(values.get('basin_days',20000.))
    workers=min(int(values.get('basin_workers',4)), os.cpu_count() or 1,n)
    seed=int(values.get('basin_seed',42))
    if not (n>=16 and days>=500 and workers>=1 and seed>=0): raise ValueError('Invalid basin sampling settings')
    eq=solve_equilibria(p)
    upper_s=float(values.get('basin_S_max',0.))
    if upper_s==0:
        s=seagrass_biomass(0.,0.,p)
        upper_s=1.25*max([1., *[e.biomass[1] for e in eq], s if math.isfinite(s) else 0.])
    if not math.isfinite(upper_s) or upper_s<=0: raise ValueError('S_max must be positive and finite')
    ctx.checkpoint(1,'Discovering stable attractors',force=True)
    enabled=values.get('basin_cycles',True) in (True,'true','True',1,'1')
    target_start=time.monotonic()
    targets, discovery=discover_targets(p,eq,enabled,lambda msg:ctx.checkpoint(2,'Cycle discovery · '+msg))
    target_seconds=time.monotonic()-target_start
    (ctx.folder/'basin_targets.json').write_text(json.dumps(dict(targets=targets,cycle_discovery=discovery),indent=2,allow_nan=False),encoding='utf-8')
    adaptive=str(values.get('basin_sampling_mode','adaptive'))=='adaptive'
    mode=str(values.get('basin_backend','fast'))
    budget=int(values.get('basin_boundary_budget',128)); depth=int(values.get('basin_boundary_depth',5))
    compare=values.get('basin_compare_reference',False) in (True,'true','True',1,'1')
    if mode not in ('fast','reference') or budget<18 or depth<1:raise ValueError('Invalid basin backend or refinement settings')
    maximum_work=n+(budget if adaptive else 0)+(n if compare else 0)
    done_before=0; sample_start=time.monotonic()
    def checkpoint(done,stage,seconds):
        ctx.checkpoint(5+85*min(1.,(done_before+done)/maximum_work),f'{stage} · {done_before+done} integrations · {seconds:.0f} s')
    sampling={}; ids=np.full(n,-1,int)
    with BasinPool(p,targets,days,workers,mode,checkpoint) as pool:
        if adaptive:
            states,result_rows,weights,ids,sampling=adaptive_samples(p,upper_s,n,budget,depth,seed,pool.evaluate)
        else:
            states=sample_initial(p,upper_s,n,seed)
            result_rows=pool.evaluate(states,'Uniform basin integrations')
            weights=np.full(n,1./n)
        done_before=pool.done
    sampling_seconds=time.monotonic()-sample_start
    results=dict(enumerate(result_rows));labels=np.array([r['label'] for r in result_rows])
    area_states=states*np.array([1.,1.,p.H]); volume=.5*p.N_wc**2*upper_s
    rows=weighted_fractions(labels,targets,volume,weights) if adaptive else fractions(labels,targets,volume)
    comparison={}
    if compare:
        ref_start=time.monotonic()
        with BasinPool(p,targets,days,workers,'reference',checkpoint) as pool:
            reference=pool.evaluate(states,'Reference verification')
        ref_seconds=time.monotonic()-ref_start
        ref_labels=np.array([r['label'] for r in reference]); different=labels!=ref_labels
        comparison=dict(reference_seconds=ref_seconds,disagreement_count=int(different.sum()),
                        disagreement_weight=float(weights[different].sum()),
                        resolved_label_conflicts=int(((labels>=0)&(ref_labels>=0)&different).sum()))
        export_rows(ctx.folder/'basin_comparison.csv',[dict(index=i,fast_label=int(labels[i]),reference_label=int(ref_labels[i]),
            agree=bool(not different[i]),weight=float(weights[i]),fast_days=result_rows[i]['days'],reference_days=reference[i]['days']) for i in range(n)])
        # Benchmark only measures audit time directly in uniform mode; pilot work is extra.
        if not adaptive:comparison['audit_speedup']=ref_seconds/max(sampling_seconds,1e-9)
    if adaptive:
        (ctx.folder/'basin_sampling.json').write_text(json.dumps(sampling,indent=2,allow_nan=False),encoding='utf-8')
    export_rows(ctx.folder/'basin_fractions.csv',rows)
    export_rows(ctx.folder/'basin_samples.csv',[dict(index=i,M0=float(a[0]),S0=float(a[1]),P0_area=float(a[2]),
        attractor_index=int(labels[i]),attractor=targets[labels[i]]['id'] if labels[i]>=0 else 'Unresolved',
        days=results[i]['days'],reason=results[i]['reason'],weight=float(weights[i]),cell=int(ids[i]),backend=results[i]['backend']) for i,a in enumerate(area_states)])
    summary=dict(seconds=time.monotonic()-started,N_wc=p.N_wc,H=p.H,S_max=upper_s,samples=n,seed=seed,workers=workers,
                 max_days=days,domain_volume_areal=volume,distribution='uniform on M>=0, HP>=0, M+HP<=N_wc, 0<=S<=S_max',
                 cycle_discovery=discovery['status'],fractions=rows,
                 sampling_mode='adaptive' if adaptive else 'uniform',pilot_count=sampling.get('pilot_count',0),
                 leaf_count=sampling.get('leaf_count',0),target_discovery_seconds=target_seconds,sampling_seconds=sampling_seconds,
                 backends=dict(Counter(r['backend'] for r in result_rows)),reference_comparison=comparison,
                 confidence_method='conditional weighted Hoeffding' if adaptive else 'Wilson',
                 convergence_days_median=float(np.median([r['days'] for r in results.values()])),
                 limitations='Finite-time numerical classification; intervals describe sampling uncertainty, not missed attractors or numerical error. At most one stable interior cycle is searched for; unresolved samples remain in denominator.')
    (ctx.folder/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    ctx.checkpoint(92,'Plotting basin fractions',force=True)
    plots(ctx.folder/'figures',p,upper_s,area_states,labels,targets,rows,adaptive=adaptive)
    if adaptive:boundary_plot(ctx.folder/'figures',p,upper_s,area_states,labels,ids,sampling,rows)
    ctx.checkpoint(100,'Completed basin analysis',force=True)


def main():
    values,ctx,_=load_args()
    try: run(values,ctx)
    except StopRequested: return 130
    return 0


def boundary_plot(folder,p,upper_s,states,labels,ids,sampling,rows):
    """Plot only tested points in candidate interface cells; no invented surface."""
    from scripts.continuation_plots import plt,save
    fig=plt.figure(figsize=(9,7));ax=fig.add_subplot(111,projection='3d')
    candidates={c['id'] for c in sampling['cells'] if c['possible_boundary']}
    mask=np.isin(ids,list(candidates))
    for row in rows:
        q=states[mask&(labels==row['index'])]
        if len(q):ax.scatter(*q.T,color=row['color'],s=17,alpha=.8,label=row['attractor'],depthshade=False)
    ax.set(xlabel='M₀ [mmol N/m²]',ylabel='S₀ [mmol N/m²]',zlabel='HP₀ [mmol N/m²]',
           xlim=(0,p.N_wc),ylim=(0,upper_s),zlim=(0,p.N_wc),
           title='Sampled candidate basin interfaces — not a reconstructed separatrix')
    if mask.any():ax.legend(fontsize=8)
    else:ax.text2D(.04,.92,'No interface detected by this finite search',transform=ax.transAxes)
    fig.tight_layout();save(fig,folder,'basin_boundary_candidates')
