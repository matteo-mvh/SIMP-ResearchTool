"""Time integration settings shared by the form and runner."""
import math
from model.parameter_metadata import parameter_schema
from model.shared_eqn import Parameters


def job_schema():
    schema=parameter_schema()
    schema.update({
        'depth_count':dict(type='choice',default='1',options=['1','2','4'],description='Depth panels'),
        'depths':dict(type='string',default='4',description='Depths (m), comma separated'),
        'duration':dict(type='float',default=1000.,min=.001,description='Simulation duration (days)'),
        'samples':dict(type='integer',default=1001,min=2,max=100000,description='Saved time samples'),
        'plot_groups':dict(type='string',default='MS, P, N',description='Plot groups: e.g. MS, S, P (letters M, S, P, N)'),
        'initial_M':dict(type='float',default=1.,min=0,description='Initial M (mmol N/m²)'),
        'initial_S':dict(type='float',default=1.,min=0,description='Initial S (mmol N/m²)'),
        'initial_P':dict(type='float',default=1.,min=0,description='Initial P (mmol N/m²)'),
    })
    return schema


def parse_job(values):
    v={k:m['default'] for k,m in job_schema().items()};v.update(values)
    depths=tuple(float(x.strip()) for x in str(v['depths']).split(',') if x.strip())
    if int(v['depth_count']) not in (1,2,4) or len(depths)!=int(v['depth_count']):
        raise ValueError('Enter 1, 2, or 4 depths matching the selected panel count')
    if len(set(depths))!=len(depths) or any(not math.isfinite(h) or h<=0 for h in depths):
        raise ValueError('Depths must be distinct positive finite numbers')
    groups=[g.strip().upper() for g in str(v['plot_groups']).split(',')]
    if not groups or any(not g or set(g)-set('MSPN') or len(set(g))!=len(g) for g in groups):
        raise ValueError('Use comma-separated groups of M, S, P, N, e.g. MS, S, P')
    if not math.isfinite(float(v['duration'])) or float(v['duration'])<=0:
        raise ValueError('Duration must be finite and positive')
    if not 2<=int(v['samples'])<=100000: raise ValueError('Samples must be between 2 and 100000')
    initial=[float(v['initial_'+k]) for k in 'MSP']
    if any(not math.isfinite(x) or x<0 for x in initial):raise ValueError('Initial biomasses must be finite and nonnegative')
    p=Parameters.from_mapping({**v,'H':depths[0]},allow_extra=True)
    if initial[0]+initial[2]>p.N_wc:
        raise ValueError('Initial M + P must not exceed the water-column inventory N_wc (areal units)')
    return p,depths,groups,initial,v
