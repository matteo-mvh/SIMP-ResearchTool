"""Disk-backed LHS, bounded parallel equilibrium screening and streaming seeds."""
import csv
import json
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from dataclasses import asdict
from pathlib import Path
import numpy as np
from model.shared_eqn import Parameters,solve_equilibria
from model.sensitivity_config import parse_job
from app.core.control import load_args,StopRequested


def lhs_file(path,n,ranges,seed):
    """One random permutation per column: exactly one draw in each LHS stratum.

    Disk-backed matrix; temporary memory is O(n), not O(n * parameters).
    """
    keys=list(ranges)
    x=np.lib.format.open_memmap(path,mode='w+',dtype='float64',shape=(n,len(keys)))
    rng=np.random.default_rng(seed)
    for j,key in enumerate(keys):
        lo,hi=ranges[key];u=(rng.permutation(n)+rng.random(n))/n
        x[:,j]=lo+u*(hi-lo) if key not in ('gamma','beta') else np.minimum(hi,lo+np.floor(u*(hi-lo+1)))
    x.flush();return x


def evaluate(index,values,grid):
    try:
        eq=solve_equilibria(Parameters(**values),grid_points=grid)
        stable=[e for e in eq if e.stability.classification=='stable']
        marginal=sum(e.stability.classification=='nonhyperbolic' for e in eq)
        # Two strict stable states prove detected bistability even with marginal extras.
        status='ok' if not marginal or len(stable)>=2 else 'marginal'
        s=[e.biomass[1] for e in stable]
        delta=max(s)-min(s) if len(s)>=2 else None
        states=[{'community':e.community,'M':e.biomass[0],'S':e.biomass[1],
                 'P_area':e.biomass[2]*values['H'],'N_area':e.nitrogen*values['H'],
                 'max_real_eigenvalue':e.stability.spectral_abscissa} for e in stable]
        return index,values,status,int(len(stable)>=2) if status=='ok' else None,delta,len(stable),marginal,states,''
    except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as exc:
        return index,values,'error',None,None,0,0,[],str(exc)


def batch_evaluate(batch,grid):return [evaluate(i,v,grid) for i,v in batch]


def run(values,ctx,job_id):
    ctx.checkpoint(0,'Preparing LHS samples',force=True)
    base,ranges,v=parse_job(values);base=asdict(base);n=int(v['sample_count']);keys=list(ranges)
    x=lhs_file(ctx.folder/'lhs.npy',n,ranges,int(v['random_seed']))
    outputs=np.lib.format.open_memmap(ctx.folder/'metrics.npy',mode='w+',dtype='float64',shape=(n,4))
    outputs[:]=np.nan  # columns: bistability, delta S, stable count, marginal count
    workers=int(v['workers']);grid=257 if v['accuracy']=='Standard' else 513
    started=time.monotonic();done=0;errors=0;marginals=0
    rng=np.random.default_rng(int(v['random_seed'])+1)
    reservoir=rng.choice(n,size=min(n,20000),replace=False).tolist()
    def batches():
        for start in range(0,n,16):
            yield [(i,{**base,**dict(zip(keys,map(float,x[i])))}) for i in range(start,min(n,start+16))]
    fields=['seed_id',*base,'metric__bistability','metric__delta_S','metric__stable_count','metric__marginal_count','metric__solver_status']
    pool=None
    try:
        with (ctx.folder/'seeds.csv').open('w',newline='',encoding='utf-8') as sf, (ctx.folder/'stable_states.jsonl').open('w',encoding='utf-8') as ef:
            writer=csv.DictWriter(sf,fieldnames=fields);writer.writeheader()
            def record(results):
                nonlocal done,errors,marginals
                for i,p,status,b,delta,count,margin,states,error in results:
                    ident=f'S-{job_id}-{i+1:09d}'
                    writer.writerow({'seed_id':ident,**p,'metric__bistability':b,'metric__delta_S':delta,
                                     'metric__stable_count':count,'metric__marginal_count':margin,'metric__solver_status':status})
                    ef.write(json.dumps({'seed_id':ident,'stable_states':states,'status':status,'error':error})+'\n')
                    outputs[i]=[np.nan if b is None else b,np.nan if delta is None else delta,count,margin]
                    errors+=status=='error';marginals+=status=='marginal';done+=1
                ctx.checkpoint(85*done/n,f'Equilibria {done:,}/{n:,} · {workers} workers')
            if workers==1:
                for batch in batches():record(batch_evaluate(batch,grid))
            else:
                if os.name=='nt':
                    # Bypass the venv redirector: it would introduce another process
                    # and break multiprocessing's parent/pipe handle handoff.
                    os.environ['__PYVENV_LAUNCHER__']=str(Path(sys.executable).with_name('pythonw.exe'))
                    mp.set_executable(str(Path(sys._base_executable).with_name('pythonw.exe')))
                pool=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'))
                iterator=iter(batches());pending=set()
                for _ in range(workers*2):
                    batch=next(iterator,None)
                    if batch is not None:pending.add(pool.submit(batch_evaluate,batch,grid))
                while pending:
                    finished,pending=wait(pending,timeout=.3,return_when=FIRST_COMPLETED)
                    ctx.checkpoint(85*done/n,f'Equilibria {done:,}/{n:,} · {workers} workers')
                    for future in finished:
                        record(future.result())
                        batch=next(iterator,None)
                        if batch is not None:pending.add(pool.submit(batch_evaluate,batch,grid))
        outputs.flush()
    finally:
        if pool:pool.shutdown(wait=True,cancel_futures=True)
    eligible=int(np.count_nonzero(np.isfinite(outputs[:,0])))
    bistable=int(np.nansum(outputs[:,0]))
    summary={'sample_count':n,'eligible_samples':eligible,'bistable_samples':bistable,'solver_errors':errors,
             'marginal_samples':marginals,'bistability_probability':bistable/eligible if eligible else None,
             'bistability_fraction_bounds':[bistable/n,(bistable+errors+marginals)/n],
             'delta_S_definition':'max(S) - min(S) among strictly stable equilibria, mmol N/m²; undefined otherwise',
             'sampling':'Independent uniform randomized LHS; grazing switches discrete uniform',
             'limitations':'Detected equilibrium bistability, not basin probability. Finite root search may miss roots. Marginal/error samples excluded from probability denominator. Bounds only account for flagged samples, not missed roots.',
             'compute_seconds':time.monotonic()-started,'ranges':ranges,'plot_sample_size':len(reservoir)}
    ctx.checkpoint(85,'Plotting sensitivity diagnostics',force=True)
    from scripts.sensitivity_plots import plot_results
    plot_results(x,outputs,keys,ranges,reservoir,summary,ctx,int(v['random_seed']))
    (ctx.folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    ctx.checkpoint(100,'Completed LHS sensitivity',force=True)


def main():
    values,ctx,job=load_args()
    try:run(values,ctx,job['id'])
    except StopRequested:return 130
    return 0
