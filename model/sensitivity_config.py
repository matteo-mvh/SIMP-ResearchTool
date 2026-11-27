"""Independent uniform LHS ranges; defaults come from the shared thesis model."""
import json
import math
import os
from dataclasses import asdict
from model.shared_eqn import Parameters
from model.parameter_metadata import parameter_schema


def default_ranges():
    return {k:[v*.5,v*1.5] for k,v in asdict(Parameters()).items() if k not in ('gamma','beta')}


def job_schema():
    schema=parameter_schema()
    schema.update({
        'ranges':dict(type='string',default=json.dumps(default_ranges()),description='Included parameter ranges'),
        'sample_count':dict(type='integer',default=1000,min=2,max=10000000,description='LHS parameter sets'),
        'random_seed':dict(type='integer',default=42,min=0,description='Random seed (reproducible sampling)'),
        'workers':dict(type='integer',default=max(1,min(8,(os.cpu_count() or 2)-1)),min=1,max=os.cpu_count() or 1,description='Parallel CPU workers'),
        'accuracy':dict(type='choice',default='Standard',options=['Standard','Thorough'],description='Root search resolution'),
    })
    return schema


def parse_job(values):
    v={k:m['default'] for k,m in job_schema().items()};v.update(values)
    p=Parameters.from_mapping(v,allow_extra=True)
    ranges=json.loads(v['ranges'])
    if not isinstance(ranges,dict):raise ValueError('Ranges must be a parameter mapping')
    for k,bounds in ranges.items():
        if k not in asdict(p) or len(bounds)!=2:raise ValueError(f'Invalid range: {k}')
        lo,hi=map(float,bounds)
        if not all(math.isfinite(x) for x in (lo,hi)) or not 0<=lo<hi:raise ValueError(f'{k}: require 0 ≤ minimum < maximum')
        if k in ('gamma','beta') and (lo!=int(lo) or hi!=int(hi)):raise ValueError('Grazing switches need integer endpoints')
        for endpoint in (lo,hi):Parameters(**{**asdict(p),k:endpoint})
        if k in ('k_S','m_S','m_M') and lo<=0:raise ValueError(f'{k}: reduced solver requires a positive minimum')
        ranges[k]=[lo,hi]
    lower={**asdict(p),**{k:b[0] for k,b in ranges.items()}}
    if min(lower['k_S'],lower['m_S'],lower['m_M'],lower['m_P']+lower['g_P'])<=0:
        raise ValueError('Reduced solver requires positive k_S, m_S, m_M and m_P + g_P')
    if ('gamma' in ranges and ranges['gamma'][0]==0 or p.gamma==0) and ('beta' in ranges or p.beta==2):
        raise ValueError('A range must not include the unsupported grazing pair gamma=0, beta=2')
    if not 2<=int(v['sample_count'])<=10000000:raise ValueError('Choose 2–10,000,000 samples')
    if not 1<=int(v['workers'])<= (os.cpu_count() or 1):raise ValueError('Invalid worker count')
    return p,ranges,v
