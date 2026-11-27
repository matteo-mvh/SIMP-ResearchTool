"""Offline, sampled 3D phase-space viewer for SIMP 1D continuation results.
Generate separately from continuation: no changes to numerical results.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

# Prevent oversubscription when basin trajectories are parallelized.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from model.shared_eqn import solve_equilibria, growth_expressions, seagrass_biomass
from model.continuation_config import parse_job

EQ_TOL = 2e-4
CYCLE_TOL = .025
COLORS = {'0':'#777777','M':'#df8531','S':'#258c58','P':'#418cc8','MS':'#9e9d41',
          'MP':'#ac476e','SP':'#228b86','MSP':'#8055b5'}


def atomic(path, text):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(text, encoding='utf-8')
    os.replace(temp, path)


def load(path, default):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else default


def rounded(a):
    return np.asarray(a, float).round(7).tolist()


def area(y, H):
    return np.asarray(y)*np.array([1., 1., H])


def growth(z, p):
    # Log coordinates preserve positive populations even when extremely rare.
    y = np.exp(z)
    return np.asarray(growth_expressions(*y, p, np.exp), float)


def physical(y, p):
    return (np.all(np.isfinite(y)) and np.min(y) >= 0 and
            np.max(y[0]+p.H*y[2]-p.N_wc) <= 1e-7*max(1., p.N_wc))


def initial_grid(p, upper_s, n):
    """Interior cell centres in M+H*P<N_wc; S is independent of N_wc."""
    for k in range(n):
        S = upper_s*(k+.5)/n
        for j in range(n):
            for i in range(n-j):
                # Strictly positive and strictly below the nitrogen boundary.
                M = p.N_wc*(i+1/3)/n
                P = p.N_wc*(j+1/3)/(n*p.H)
                if min(M, S, P)>0:
                    yield (M, S, P), k, j


def cycle_matches(row, result):
    """Do not silently replace a saved cycle with a different orbit."""
    if not result:
        return False
    r = result['row']
    if abs(math.log(r['period']/float(row['period']))) > .02:
        return False
    for key in ('M', 'S', 'P_area'):
        width = max(float(row[key+'_max'])-float(row[key+'_min']), .001)
        for field in ('min', 'max', 'mean'):
            if abs(r[key+'_'+field]-float(row[key+'_'+field])) > .08*width+1e-5:
                return False
    return True


def recover_cycle(p, row, equilibria, previous=None):
    """Reverify stored phase seeds; older results require orbit reconstruction."""
    from scripts.continuation_runner import trajectory_guess, correct_cycle, verify_cycle
    T = float(row['period'])
    if row.get('cycle_log_state') is not None:
        seed = dict(log_state=np.asarray(row['cycle_log_state']), period=T,
                    method='saved_phase_seed')
        result = verify_cycle(p, seed)
        if cycle_matches(row, result):
            return result['seed']
    mean = np.array([row['M_mean'], row['S_mean'], row['P_area_mean']/p.H])
    seeds = []
    if previous is not None:
        seeds.append(np.exp(previous['log_state']))
    seeds.extend([mean*np.array([1.1,.9,1.15]), mean*np.array([.9,1.1,.85]),
                  np.array([row['M_mean'], row['S_mean'], row['P_area_max']/p.H])])
    for state in seeds:
        try:
            guess = trajectory_guess(p, state, equilibria, period_hint=T)
            if guess is None:
                continue
            result = verify_cycle(p, dict(guess, method='reconstructed_return')) if guess.get('converged') else None
            if cycle_matches(row, result):
                return result['seed']
            corrected = correct_cycle(p, guess)
            if corrected is not None:
                result = verify_cycle(p, corrected)
                if cycle_matches(row, result):
                    return result['seed']
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError):
            continue
    return None


def orbit_samples(p, seed):
    T = float(seed['period'])
    sol = solve_ivp(lambda t,z:growth(z,p), (0,T), seed['log_state'], method='LSODA',
                    rtol=1e-10, atol=1e-12, dense_output=True, max_step=T/80)
    if not sol.success or not physical(np.exp(sol.y), p):
        return None
    if np.max(abs(sol.y[:,-1]-sol.y[:,0])) > 5e-6:
        return None
    return area(np.exp(sol.sol(np.linspace(0,T,513))).T,p.H)


def make_targets(p, equilibria, orbit, period):
    targets = []
    seen = {}
    for e in sorted(equilibria,key=lambda e:(e.community,e.biomass)):
        seen[e.community] = seen.get(e.community,0)+1
        ident = e.community+(' '+str(seen[e.community]) if seen[e.community]>1 else '')
        targets.append(dict(id=ident, state=area(e.biomass,p.H).tolist(),
                            stable=e.stability.classification, kind='equilibrium',
                            color=COLORS[e.community] if seen[e.community]==1 else
                            ['#d04177','#0d849b','#9e5b21','#6073af'][((seen[e.community]-2)%4)]))
    if orbit is not None:
        targets.append(dict(id='Verified cycle', kind='cycle', stable='stable',
                            color='#c03985', path=rounded(orbit), period=period))
    return targets


# Per-frame data is sent once to each process, not for every integration.
_CONTEXT = None


def init_worker(p, targets, duration):
    global _CONTEXT
    eq = [(i,np.array(t['state']),np.maximum(abs(np.array(t['state'])),1.))
          for i,t in enumerate(targets) if t['kind']=='equilibrium' and t['stable']=='stable']
    cycles = []
    for i,t in enumerate(targets):
        if t['kind'] != 'cycle':
            continue
        path = np.asarray(t['path'])
        scale = np.maximum(np.ptp(path,axis=0), 1e-4*np.maximum(np.mean(path,axis=0),1.))
        cycles.append((i,float(t['period']),scale,cKDTree(path/scale)))
    _CONTEXT = p,eq,cycles,duration


def classify_seed(item):
    p,eq,cycles,max_days = _CONTEXT
    y0,k,j = item
    start = area(y0,p.H)
    z = np.log(y0)
    chunks, elapsed, label, reason = [],0.,-1,'integration horizon reached'
    # Sample three successive periods to distinguish recurrence from a
    # transient passage near a cycle. No label from extrema or a single endpoint.
    interval = max(500., max((4*T for _,T,_,_ in cycles),default=0.))
    try:
        while elapsed < max_days:
            dt = min(interval,max_days-elapsed)
            sol = solve_ivp(lambda t,z:growth(z,p),(0.,dt),z,method='LSODA',
                            rtol=2e-8,atol=2e-10,dense_output=True)
            if not sol.success or not physical(np.exp(sol.y),p):
                reason = 'integration failed or left physical domain'
                break
            z = sol.y[:,-1]
            elapsed += dt
            chunks.append(area(np.exp(sol.sol(np.linspace(0,dt,100))).T,p.H))
            found = []
            tail = area(np.exp(sol.sol(np.linspace(max(0.,dt-min(250.,dt)),dt,8))).T,p.H)
            for i,point,scale in eq:
                err = float(np.max(abs(tail-point)/scale))
                # Require all terminal samples to remain close, plus a small
                # vector field at the endpoint; local stability was checked separately.
                speed = np.max(abs(tail[-1]*growth(z,p))/scale)
                if err <= EQ_TOL and speed <= 1e-6:
                    found.append((i,'stable equilibrium: persistent proximity'))
            for i,T,scale,tree in cycles:
                if dt < 3*T:
                    continue
                ts = np.linspace(dt-T,dt,65)
                now = area(np.exp(sol.sol(ts)).T,p.H)
                old = area(np.exp(sol.sol(ts-T)).T,p.H)
                older = area(np.exp(sol.sol(ts-2*T)).T,p.H)
                distance = max(float(tree.query(a/scale)[0].max()) for a in (now,old,older))
                recurrence = max(float(np.max(abs(now-old)/scale)),float(np.max(abs(old-older)/scale)))
                amplitude = np.max(np.ptp(now,axis=0)/scale)
                if distance <= CYCLE_TOL and recurrence <= CYCLE_TOL and amplitude >= .8:
                    found.append((i,'verified cycle: three recurrent periods'))
            if len(found)==1:
                label,reason = found[0]
                break
            if len(found)>1:
                reason = 'ambiguous attractor assignment'
            interval = min(2*interval,max_days)
    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        reason = str(exc)
    path = np.concatenate(chunks) if chunks else start[None,:]
    if len(path)>180:
        indices = np.linspace(0,len(path)-1,180).astype(int)
        path = path[indices]
    return dict(initial=rounded(start),s_layer=k,p_layer=j,label=label,reason=reason,
                days=round(elapsed,2),path=rounded(path))


def choose_frames(axis, locations, events, count):
    # Use actual completed continuation coordinates, include verified Hopf
    # points, and prioritize all verified periodic locations. No interpolation.
    existing = {float(r['u']):r for r in locations if 'u' in r}
    grid = list(np.linspace(0,1,count))
    chosen = set(round(float(u),10) for u in grid)
    if existing:
        chosen = {min(existing,key=lambda v:abs(v-u)) for u in chosen}
    chosen.update(u for u,r in existing.items() if r.get('stable_cycle_found'))
    chosen.update(round(float(e['u']),10) for e in events if e.get('kind')=='hopf'
                  and e.get('classification_status')=='verified')
    chosen.update((0.,1.))
    return sorted(u for u in chosen if 0<=u<=1),existing


def build(folder, n=5, frames=21, days=20000., workers=2):
    config = load(folder/'config.json',{})
    if config.get('analysis_id')!='continuation_1d':
        raise ValueError('Select a 1D continuation result folder')
    base, depths, axes, _ = parse_job(config['parameters'],1)
    axis = axes[0]
    output = folder/'figures'/'phase_space_3d.html'
    output.parent.mkdir(exist_ok=True)
    progress = folder/'phase_space_progress.json'
    started = time.monotonic()
    from model.parameter_metadata import parameter_schema
    data = dict(parameter=axis.name,parameter_unit=parameter_schema()[axis.name]['unit'],units='mmol N m⁻²',depths=[],grid=n,days=days,
                model='SIMP',eq_tol=EQ_TOL,cycle_tol=CYCLE_TOL)
    for H in depths:
        suffix=f'H{H:g}'
        payload=load(folder/f'periodic_{suffix}.json',{})
        locations=payload.get('locations',[])
        import csv
        eventfile=folder/'figures_revised'/f'bifurcations_{suffix}.csv'
        if not eventfile.is_file():
            eventfile=folder/f'bifurcations_{suffix}.csv'
        events=[]
        if eventfile.is_file():
            with eventfile.open(encoding='utf-8-sig',newline='') as f:
                events=list(csv.DictReader(f))
        coords, lookup=choose_frames(axis,locations,events,frames)
        params=[replace(base,H=H,**{axis.name:axis.physical(u)}) for u in coords]
        equilibria=[solve_equilibria(p) for p in params]
        # Consistent physical axis scales throughout the slider. Initial S is
        # sampled to 1.25 times the maximum possible unshaded equilibrium.
        s_values=[float(e.biomass[1]) for eq in equilibria for e in eq]
        for p in params:
            val=seagrass_biomass(0.,0.,p)
            if math.isfinite(val): s_values.append(val)
        upper_s=1.25*max([1.,*s_values])
        item=dict(H=H,frames=[],initial_s_upper=upper_s,bounds=[max(1.,max(p.N_wc for p in params)),upper_s,
                                       max(1.,max(p.N_wc for p in params))])
        data['depths'].append(item)
        previous=None
        for index,(u,p,eq) in enumerate(zip(coords,params,equilibria)):
            note=f'H={H:g} m · frame {index+1}/{len(coords)} · {axis.name}={axis.physical(u):.6g}'
            def report(stage):
                message=dict(stage=stage,detail=note,elapsed_seconds=round(time.monotonic()-started,1))
                atomic(progress,json.dumps(message))
                print(note+' · '+stage,flush=True)
            report('Preparing attractors')
            row=lookup.get(u,{})
            cycle,period=None,None
            verified_saved=bool(row.get('stable_cycle_found') and row.get('status')=='verified_periodic'
                                and row.get('stability')=='stable')
            if verified_saved:
                try:
                    seed=recover_cycle(p,row,eq,previous)
                    if seed is not None:
                        cycle=orbit_samples(p,seed)
                        if cycle is not None:
                            previous,period=seed,float(seed['period'])
                except (ValueError, RuntimeError, ArithmeticError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
                    print('Cycle geometry unavailable:', exc, flush=True)
            targets=make_targets(p,eq,cycle,period)
            for target in targets:
                maximum = np.max(target['path'],axis=0) if target['kind']=='cycle' else target['state']
                item['bounds']=np.maximum(item['bounds'],maximum).tolist()
            seeds=list(initial_grid(p,upper_s,n))
            samples=[]
            report(f'Integrating {len(seeds)} initial conditions')
            if workers>1 and seeds:
                with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,
                                         initargs=(p,targets,days)) as pool:
                    for sample in pool.map(classify_seed,seeds,chunksize=2):
                        samples.append(sample)
                        if len(samples)%20==0: report(f'Basin samples {len(samples)}/{len(seeds)}')
            else:
                init_worker(p,targets,days)
                for seed in seeds:
                    samples.append(classify_seed(seed))
                    if len(samples)%20==0: report(f'Basin samples {len(samples)}/{len(seeds)}')
            # Include actual sampled trajectories in global display bounds.
            for sample in samples:
                if sample['path']:
                    item['bounds']=np.maximum(item['bounds'],np.max(sample['path'],axis=0)).tolist()
            item['frames'].append(dict(x=float(axis.physical(u)),u=u,N_wc=p.N_wc,
                targets=targets,samples=samples,cycle_status=('verified geometry' if cycle is not None else
                    'saved cycle; geometry not recovered' if verified_saved else 'no verified cycle geometry'),
                events=[e.get('classification') for e in events if abs(float(e['u'])-u)<1e-8]))
    html=Path(__file__).with_name('phase_space_viewer.html').read_text(encoding='utf-8')
    atomic(output,html.replace('__SIMP_DATA__',json.dumps(data,separators=(',',':'),allow_nan=False).replace('</','<\\/')))
    atomic(progress,json.dumps(dict(stage='Completed',detail=str(output),elapsed_seconds=round(time.monotonic()-started,1))))
    print(output,flush=True)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--grid',type=int,default=5,choices=range(3,13))
    parser.add_argument('--frames',type=int,default=21,choices=range(5,101))
    parser.add_argument('--days',type=float,default=20000.)
    parser.add_argument('--workers',type=int,default=2,choices=range(1,9))
    args=parser.parse_args()
    if not 100<=args.days<=1000000: parser.error('--days must be between 100 and 1000000')
    build(args.folder,args.grid,args.frames,args.days,args.workers)


if __name__=='__main__':
    main()
