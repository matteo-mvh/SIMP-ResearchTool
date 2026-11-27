"""Small empirical runtime model; no scientific dependencies in the GUI."""
import json
import math
import statistics
import time
from pathlib import Path
from datetime import datetime


def measured_seconds(job):
    try:
        seconds = (datetime.fromisoformat(job['ended_at']) -
                   datetime.fromisoformat(job['started_at'])).total_seconds()
        return seconds if math.isfinite(seconds) and seconds > 0 else None
    except (ValueError, TypeError, KeyError):
        return None


def workload(parameters, analysis):
    def number(key, default):
        try: return max(1, float(parameters.get(key, default)))
        except (TypeError, ValueError): return default
    if analysis in ('continuation_1d', 'continuation_2d'):
        dimensions = 2 if analysis.endswith('2d') else 1
        # Discovery is a grid in 2D. Tracing and rendering also contribute.
        discovery = (number('discovery_points', 9) / 9) ** dimensions
        tracing = number('max_steps', 600) / 600
        resolution = number('x_resolution', 101) / 101
        if dimensions == 2: resolution *= number('y_resolution', 101) / 101
        refinement=1.
        if dimensions==2:
            refinement+=.2*max(0,float(parameters.get('refinement_rounds',0)))*number('refinement_searches',12)/12
        return number('depth_count', 1) * (.4*discovery + .4*tracing + .2*resolution)*refinement
    if analysis == 'timeseries':
        return number('depth_count',1) * (number('duration',1000)/1000 + number('samples',1001)/1001)
    if analysis == 'sensitivity':
        return number('sample_count',1000)/1000 / number('workers',1) * (2 if parameters.get('accuracy')=='Thorough' else 1)
    for key in ('sample_count', 'points', 'days'):
        if key in parameters: return number(key, 1)
    return 1


def estimate(job, history):
    parameters = json.loads(job['config_json']).get('parameters', {})
    analysis = job['analysis_id']
    candidates = []
    for row in history:
        old = json.loads(row['parameters_json'])
        distance = 0
        for key in parameters.keys() | old.keys():
            a, b = parameters.get(key), old.get(key)
            if a == b: continue
            if isinstance(a, (float, int)) and isinstance(b, (float, int)):
                distance += abs(a-b) / max(abs(a), abs(b), 1e-9)
            elif key != 'seed_id': distance += 1
        seconds = float(row['seconds']) * workload(parameters, analysis) / workload(old, analysis)
        if math.isfinite(seconds) and seconds > 0:
            candidates.append((distance, seconds))
    if not candidates: return None
    candidates.sort()
    nearest = candidates[:5]
    return statistics.median(s for _, s in nearest), len(nearest)


def duration(seconds):
    if seconds < 60: return f'{max(1, round(seconds))} s'
    if seconds < 3600: return f'{round(seconds/60)} min'
    return f'{seconds/3600:.1f} h'


def estimate_label(job, prediction, paused=False):
    if paused: return 'Paused · time estimate resumes with the run'
    if prediction is None: return 'Time estimate: learning from completed runs'
    total, count = prediction
    # Progress measures stages rather than wall time, so this is approximate.
    fraction = min(.99, max(0, float(job.get('progress') or 0)/100))
    remaining = total * (1-fraction)
    return f'About {duration(remaining)} remaining · {duration(total)} total · based on {count} similar run(s)'


def plotting_work(parameters, analysis):
    if analysis=='sensitivity':return 5+len(json.loads(parameters.get('ranges','{}')))*2
    depths=max(1,int(parameters.get('depth_count',1)))
    if analysis=='timeseries':
        figures=len(str(parameters.get('plot_groups','MS, P, N')).split(','))
    else:figures=depths+1 if analysis=='continuation_1d' else 3
    return figures*(1+.25*(depths-1))


def live_label(job, history, paused=False):
    """Stage-specific historical timing plus observed stage progress, with cold start."""
    parameters=json.loads(job['config_json']).get('parameters',{})
    analysis=job['analysis_id']
    totals={'compute':(15 if analysis=='timeseries' else 60 if analysis=='continuation_1d' else 180)
                      *workload(parameters,analysis),
            'plot':3*plotting_work(parameters,analysis)}
    learned=0
    for stage in totals:
        samples=[]
        for row in history:
            stages=json.loads(row.get('stages_json') or '{}')
            seconds=stages.get(stage,0)
            if not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<=0:continue
            old=json.loads(row['parameters_json'])
            scale=plotting_work if stage=='plot' else workload
            samples.append(seconds*scale(parameters,analysis)/scale(old,analysis))
        if samples:
            totals[stage]=statistics.median(samples[:10]);learned=max(learned,len(samples[:10]))
    # Older successful runs still inform computation before stage telemetry exists.
    if not learned:
        previous=estimate(job,history)
        if previous:totals['compute']=max(1,previous[0]-totals['plot'])
    try:data=json.loads((Path(job['output_dir'])/'progress.json').read_text())
    except (KeyError,OSError,ValueError):data={}
    stage=data.get('stage','compute')
    if stage not in totals:stage='plot' if float(data.get('progress',0))>=85 else 'compute'
    fraction=min(1,max(0,float(data.get('stage_progress',0))))
    elapsed=float(data.get('stage_seconds',{}).get(stage,0))
    if data and not paused and job.get('status')=='running':
        elapsed+=max(0,time.time()-float(data.get('updated',time.time())))
    predicted=totals[stage]
    if fraction>.01:
        # Increasing weight on measured speed as the current stage progresses.
        weight=min(.85,fraction)
        predicted=(1-weight)*predicted+weight*elapsed/fraction
    remaining=max(1,predicted-elapsed, predicted*(1-fraction)*.1)
    if stage=='compute':remaining+=totals['plot']
    if job.get('status') in ('draft','queued'):remaining=sum(totals.values())
    source=f'learned stage timings ({learned} runs)' if learned else 'provisional; refining live'
    prefix='Paused · ' if paused else ''
    suffix=' after resume' if paused else ''
    return f'{prefix}ETA ≈ {duration(remaining)}{suffix} · {"plotting" if stage=="plot" else "calculation + plotting"} · {source}'
