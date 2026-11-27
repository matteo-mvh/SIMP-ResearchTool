"""MSP-only discovery, worker isolation and branch reuse for periodic BVPs."""
from dataclasses import asdict,replace
from pathlib import Path
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from uuid import uuid4
import numpy as np
from model.shared_eqn import solve_equilibria
from model.periodic_config import periodic_options
from model.periodic_math import admissible_msp,absence_certificate,deterministic_guesses,same_orbit,orbit_signature


def parameters_at(base,axes,uv):
    return replace(base,**{a.name:float(a.physical(v)) for a,v in zip(axes,uv)})


def equilibrium_evidence(p,known=None):
    """Separate a proof of absence from an empty/failed numerical search."""
    if known is not None:
        msp=[r for r in known if admissible_msp(r,p)]
        if msp:return dict(status='msp_present',roots=known,msp=msp,source='reused_equilibria')
    certificate=absence_certificate(p)
    if certificate:return dict(status='msp_absent_certified',roots=known or [],msp=[],reason=certificate)
    try:
        roots=[]
        for e in solve_equilibria(p,grid_points=513):
            roots.append(dict(community=e.community,stability=e.stability.classification,
                              **dict(zip('MSP',e.biomass))))
        msp=[r for r in roots if admissible_msp(r,p)]
        return dict(status='msp_present' if msp else 'msp_not_found',roots=roots,msp=msp,
                    source='reduced_solver',reason='' if msp else 'Finite equilibrium search is inconclusive; absence not certified')
    except (ValueError,RuntimeError,ArithmeticError,np.linalg.LinAlgError) as exc:
        return dict(status='equilibrium_discovery_failed',roots=known or [],msp=[],reason=str(exc))


def run_isolated(task,base,axes,options,folder,checkpoint,on_event=lambda _:None):
    """One attempt/continuation arm per process; flush points before native crashes."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    request=folder/f'request_{uuid4().hex}.json'
    request.write_text(json.dumps(dict(task=task,base=asdict(base),axes=[asdict(a) for a in axes],options=options)),encoding='utf-8')
    env=os.environ.copy()
    env.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS')})
    cwd=Path(__file__).resolve().parents[1]
    events=[];position=0;pending='';started=time.monotonic();process=None;reason='completed';covered=False
    def read_events():
        nonlocal position,pending,covered
        path=request.with_suffix('.jsonl')
        if not path.exists():return
        with path.open(encoding='utf-8') as stream:
            stream.seek(position);pending+=stream.read();position=stream.tell()
        lines=pending.split('\n');pending=lines.pop()
        for line in lines:
            if not line:continue
            item=json.loads(line);events.append(item)
            if not covered and on_event(item) is False:covered=True
    with request.with_suffix('.log').open('w',encoding='utf-8') as log:
        try:
            process=subprocess.Popen([sys.executable,'-m','model.periodic_worker',str(request.resolve())],
                cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            while process.poll() is None:
                read_events()
                if covered:
                    reason='covered_branch';process.kill();break
                checkpoint(f'Periodic {task["kind"]} · {time.monotonic()-started:.0f} s · {len(events)} records')
                if time.monotonic()-started>options['periodic_timeout']:
                    reason='worker_timeout';process.kill();break
                time.sleep(.2)
            process.wait();read_events()
            if reason=='completed' and not any(e['type']=='complete' for e in events):
                reason='solver_failed' if any(e['type']=='failure' for e in events) else 'native_worker_crashed'
        finally:
            if process is not None and process.poll() is None:
                process.kill();process.wait(timeout=5)
    return dict(reason=reason,exit_code=process.returncode,events=events,seconds=time.monotonic()-started)


def _key(uv):return tuple(round(float(v),11) for v in uv)


def matching_branch(row,branches,dimensions,tolerance=.003):
    """Compare an orbit with checked points and short continuation segments."""
    keys='uv'[:dimensions];uv=np.array([row[k] for k in keys]);sig=orbit_signature(row)
    for branch in branches:
        for old in branch['points']:
            if max(abs(row[k]-old[k]) for k in keys)<1e-5 and same_orbit(row,old):return branch['id']
        for arm in branch['arms']:
            path=[branch['points'][0],*arm['records']]
            for a,b in zip(path,path[1:]):
                start=np.array([a[k] for k in keys]);delta=np.array([b[k]-a[k] for k in keys])
                norm=float(delta@delta)
                if norm<1e-16 or norm>.03**2:continue
                fraction=float((uv-start)@delta/norm)
                if not 0<=fraction<=1 or np.linalg.norm(uv-start-fraction*delta)>1e-5:continue
                expected=(1-fraction)*orbit_signature(a)+fraction*orbit_signature(b)
                if np.max(abs(sig-expected))<tolerance:return branch['id']
    return None


def cached_equilibria(data,dimensions):
    cache={}
    for row in data.get('scan',[]):
        uv=_key([row[k] for k in 'uv'[:dimensions]])
        cache.setdefault(uv,[]).append(row)
    for path in data.get('paths',[]):
        for row in path:
            if row['community']!='MSP':continue
            uv=_key([row[k] for k in 'uv'[:dimensions]])
            roots=cache.setdefault(uv,[])
            if not any(r['community']=='MSP' and np.allclose([r[c] for c in 'MSP'],[row[c] for c in 'MSP']) for r in roots):
                roots.append(row)
    for region in data.get('regions',[]):
        for sample in region['samples']:
            if 'roots' in sample:cache[_key((sample['u'],sample['v']))]=sample['roots']
    return cache


def _hopf_seeds(data,dimensions,count):
    found=[r for r in data.get('events',[]) if r.get('kind')=='hopf' and r['community']=='MSP']
    for curve in data.get('curves',[]):
        if curve and curve[0]['kind']=='hopf' and curve[0]['community']=='MSP':
            found.extend(curve[i] for i in np.unique(np.linspace(0,len(curve)-1,min(count,len(curve)),dtype=int)))
    unique={}
    for row in found:
        entries=unique.setdefault(_key([row[k] for k in 'uv'[:dimensions]]),[])
        if not any(np.allclose([row[c] for c in 'MSP'],[old[c] for c in 'MSP'],rtol=1e-7,atol=1e-9) for old in entries):
            entries.append(row)
    return unique


def export_periodic(data,folder,depth):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    payload=data['periodic_analysis']
    path=folder/f'periodic_H{depth:g}.json'
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temporary,path)
    rows=[dict(row,branch=branch['id'],point=i) for branch in payload['branches'] for i,row in enumerate(branch['points'])]
    fields=['branch','point','status','community','stability','period','H','u','v','x','y']
    fields+=list(dict.fromkeys(k for row in rows for k in row if k not in fields))
    with (folder/f'periodic_branches_H{depth:g}.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)
    data['oscillations']=rows  # Existing consumers can still find cycle envelopes.


def analyze_periodic(data,base,axes,values,checkpoint,folder):
    options=periodic_options(values);dimensions=len(axes);cache=cached_equilibria(data,dimensions)
    payload=dict(method='pyoomph periodic boundary-value problem; no time integration',settings=options,
                 excluded_communities=['0','M','S','P','MS','MP','SP'],
                 exclusion_reason='Bendixson–Dulac exclusion for the current SIMP formulation (MSP-only search)',
                 locations=[],branches=[],hopf=[],attempts=[],coverage={})
    data['periodic_analysis']=payload;data['oscillations']=[]
    if options['periodic_analysis']=='off':
        payload['status']='disabled';export_periodic(data,folder,base.H);return
    payload['status']='running'
    hopf=_hopf_seeds(data,dimensions,options['periodic_samples'])
    # Reuse every supplied MSP equilibrium at selected coordinates, including
    # distinct coexisting MSP roots; stability never filters the search.
    grid=list(itertools.product(np.linspace(0,1,options['periodic_samples']),repeat=dimensions))
    reused=[uv for uv,roots in cache.items() if any(r['community']=='MSP' for r in roots)]
    # Stratify existing points, preserving coarse grid and Hopf coverage first.
    if reused:
        reused=[reused[i] for i in np.unique(np.linspace(0,len(reused)-1,min(len(reused),options['periodic_samples']**dimensions),dtype=int))]
    queue=list(dict.fromkeys([*hopf,*map(_key,grid),*reused]))
    scheduled=set(queue)
    attempted=set();bank=[];counter=0;refinement=[]
    def save():export_periodic(data,folder,base.H)
    def attempt(task,on_event=lambda _:None):
        nonlocal counter
        counter+=1
        directory=Path(folder)/f'periodic_workers_H{base.H:g}'/f'{counter:06d}'
        result=run_isolated(task,base,axes,options,directory,checkpoint,on_event)
        payload['attempts'].append(dict(task=counter,kind=task['kind'],uv=task.get('uv',task.get('seed',{}).get('uv')),
            mode=task.get('mode'),amplitude=task.get('amplitude'),period=task.get('period'),
            reason=result['reason'],exit_code=result['exit_code'],seconds=result['seconds'],
            failures=[e['reason'] for e in result['events'] if e['type']=='failure']))
        return result
    def accept(item,location):
        row=item['row'];seed=item['seed']
        # Only same-coordinate, same-period, same-envelope matches are duplicates.
        duplicate=matching_branch(row,payload['branches'],dimensions)
        if duplicate is not None:return duplicate
        branch=dict(id=len(payload['branches'])+1,origin=location,points=[row],arms=[])
        payload['branches'].append(branch)
        bank.append(dict(row=row,seed=seed,branch=branch['id']))
        save()
        for axis in range(dimensions):
            for direction in (-1,1):
                arm=[]
                arm_info=dict(axis=axis,direction=direction,records=arm,termination='running')
                branch['arms'].append(arm_info)
                def on_event(event):
                    if event['type']!='orbit':return
                    value=event['row']
                    old=matching_branch(value,[b for b in payload['branches'] if b['id']!=branch['id']],dimensions)
                    if old is not None:
                        arm_info['joins_branch']=old
                        return False
                    arm.append(value)
                    branch['points'].append(value)
                    bank.append(dict(row=value,seed=event['seed'],branch=branch['id']))
                    save()  # Preserve verified points even if the next native step crashes.
                result=attempt(dict(kind='trace',seed=seed,axis=axis,direction=direction),on_event)
                arm_info.update(termination=result['reason'],
                    notes=[e for e in result['events'] if e['type'] in ('termination','failure')])
        return branch['id']
    rounds=options['periodic_refinement_rounds']
    for round_index in range(rounds+1):
        this_round=[]
        for uv in queue:
            uv=_key(uv)
            if uv in attempted:continue
            if len(attempted)>=options['periodic_max_locations']:break
            attempted.add(uv)
            checkpoint(f'MSP periodic discovery · location {len(attempted)} · refinement {round_index}/{rounds}')
            evidence=equilibrium_evidence(parameters_at(base,axes,uv),cache.get(uv))
            location=dict(uv=list(uv),**{k:float(a.physical(v)) for k,a,v in zip('xy',axes,uv)},
                          equilibrium_status=evidence['status'],reason=evidence.get('reason',''),
                          msp_count=len(evidence['msp']),round=round_index,branches=[])
            payload['locations'].append(location);this_round.append(location)
            if evidence['status']!='msp_present':
                location['status']='excluded_no_msp' if evidence['status']=='msp_absent_certified' else 'equilibrium_unresolved'
                save();continue
            location['status']='orbit_discovery_unsuccessful'
            tasks=[]
            if uv in hopf:
                tasks.extend(dict(kind='hopf',equilibrium=row,axis=int(row.get('solved_axis',0))) for row in hopf[uv])
            # Neighbouring solutions are guesses, never treated as proof here.
            nearby=sorted(bank,key=lambda b:np.linalg.norm(np.asarray(b['seed']['uv'])-uv))
            used=[]
            for b in nearby:
                if b['branch'] in used:continue
                if np.linalg.norm(np.asarray(b['seed']['uv'])-uv)>.25:continue
                used.append(b['branch'])
                tasks.append(dict(kind='reuse',seed=b['seed'],equilibrium=evidence['msp'][0]))
                if len(used)>=4:break
            for root in evidence['msp']:
                tasks.extend(deterministic_guesses(root,parameters_at(base,axes,uv),options))
            for task in tasks:
                task['uv']=list(uv)
                result=attempt(task)
                for item in result['events']:
                    if item['type']=='hopf':payload['hopf'].append(item)
                    if item['type']=='orbit':
                        branch=accept(item,list(uv));location['branches'].append(branch)
                        location['status']='verified_periodic'
                save()
        # Refine near cycles, failed MSP searches and transitions in MSP existence.
        refinement=[]
        step=1/(options['periodic_samples']-1)/2**(round_index+1)
        for loc in this_round:
            if loc['equilibrium_status']=='msp_present' or loc['status']=='equilibrium_unresolved':
                for axis in range(dimensions):
                    for sign in (-1,1):
                        uv=list(loc['uv']);uv[axis]+=sign*step
                        if 0<=uv[axis]<=1 and _key(uv) not in attempted:refinement.append(_key(uv))
        queue=list(dict.fromkeys(refinement))
        if round_index<rounds:scheduled.update(queue)
    payload['coverage']=dict(checked_locations=len(attempted),max_locations=options['periodic_max_locations'],
                             unvisited_locations=[list(uv) for uv in sorted(scheduled-attempted)],
                             suggested_refinement_locations=[list(uv) for uv in queue],
                             limit_reached=bool(scheduled-attempted))
    payload['status']='completed_with_unresolved_searches' if any(l['status'] in ('equilibrium_unresolved','orbit_discovery_unsuccessful') for l in payload['locations']) else 'completed'
    save()
