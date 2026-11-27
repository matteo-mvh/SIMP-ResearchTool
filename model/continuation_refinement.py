"""Bounded conflict-driven searches; every new boundary is native pyoomph."""
from itertools import combinations
import numpy as np
from model import pyoomph_continuation as native
from model.continuation_regions import analyze_regions, JOIN_TOL, SAMPLE_MARGIN


def fingerprint(roots):
    return tuple(sorted((r['community'],r['stability']) for r in roots))


def conflict_segments(regions):
    """Nearest conflicting samples, joined by axis-aligned L-shaped routes.

    Event correction holds the other parameter fixed, so diagonal sample
    pairs must never be passed straight to the one-control corrector.
    """
    pairs=[]
    for region in regions:
        groups={}
        for s in region['samples']:groups.setdefault(s['signature'],[]).append(s)
        local=[]
        for a,b in combinations(groups.values(),2):
            pair=min(((x,y) for x in a for y in b),
                     key=lambda xy:(xy[0]['u']-xy[1]['u'])**2+(xy[0]['v']-xy[1]['v'])**2)
            local.append(pair)
        for rank,(a,b) in enumerate(local):pairs.append((rank,-region['polygon'].area,a,b))
    pairs.sort(key=lambda r:r[:2])
    for _,_,a,b in pairs:
        start=np.array([a['u'],a['v']]);end=np.array([b['u'],b['v']])
        for corner in (np.array([end[0],start[1]]),np.array([start[0],end[1]])):
            for lo,hi in ((start,corner),(corner,end)):
                if np.linalg.norm(hi-lo)>1e-7:yield lo,hi,int(np.argmax(abs(hi-lo)))


def local_transect(start,end,axis,lookup,max_checks=25):
    """Bisect changes in root multiplicity/stability before native correction."""
    points={}
    def sample(t):
        if t not in points:points[t]=lookup(start+t*(end-start))
        return points[t]
    for t in (0.,.25,.5,.75,1.):sample(t)
    for _ in range(6):
        intervals=[(a,b) for a,b in zip(sorted(points)[:-1],sorted(points)[1:])
                   if fingerprint(points[a])!=fingerprint(points[b]) and abs((b-a)*(end[axis]-start[axis]))>1e-5]
        if not intervals:break
        for a,b in intervals:
            if len(points)>=max_checks:break
            sample((a+b)/2)
        if len(points)>=max_checks:break
    return [points[t] for t in sorted(points,key=lambda t:start[axis]+t*(end[axis]-start[axis]))]


def refine_regions(dataset,base,axes,samples,max_steps,checkpoint=lambda _:None,
                   max_rounds=3,searches_per_round=12,on_round=None):
    cache={};attempted=set();retried_ends=set();history=[]
    dataset.setdefault('events',[]);dataset.setdefault('diagnostics',[]);dataset.setdefault('scan',[])
    # Initial discovery already evaluated these complete equilibrium sets.
    # Reuse them for classification instead of solving them again.
    for row in dataset['scan']:
        key=tuple(round(float(row[k]),12) for k in ('u','v'))
        rows=cache.setdefault(key,[])
        if not any(r['community']==row['community'] and np.allclose(native.state_of(r),native.state_of(row),rtol=1e-8,atol=1e-8) for r in rows):
            rows.append(row)
    def lookup(uv):
        key=tuple(round(float(t),12) for t in uv)
        if key not in cache:cache[key]=native.discover(base,axes,key,grid_points=257)
        return cache[key]
    initial_keys=set(cache)
    def classify():
        return analyze_regions(dataset,base,axes,samples,checkpoint,lookup=lookup)
    def add_event(event,round_number,allow_extension=False):
        same=[c for c in dataset['curves'] if c and c[0]['kind']==event['kind']
              and c[0]['invader']==event['invader'] and c[0]['community']==event['community']]
        if not allow_extension and native.covered(event,same,axes,tolerance=.0002,interior=True):return False
        curve,notes=native.trace_event(base,axes,event,max_steps,checkpoint)
        dataset['diagnostics'].extend(dict(n,refinement_round=round_number) for n in notes)
        if len(curve)<2:return False
        # Retrying an endpoint is useful only if it actually extends coverage.
        if all(native.covered(r,same,axes,tolerance=1e-4) for r in curve):return False
        dataset['curves'].append(curve)
        dataset['events'].append(dict(event,refinement_round=round_number))
        return True
    classify();stalled=0;stop_reason='round_limit' if max_rounds else 'refinement_disabled'
    for iteration in range(1,max_rounds+1):
        before=sum(r['polygon'].area for r in dataset['regions'] if r['label']=='?')
        if before<=1e-9:break
        searches=0;added=0;checked_before=len(cache)
        for start,end,axis in conflict_segments(dataset['regions']):
            if searches>=searches_per_round:break
            key=tuple(sorted((tuple(np.round(start,10)),tuple(np.round(end,10)))))
            if key in attempted:continue
            attempted.add(key);searches+=1
            checkpoint(f'Refinement {iteration}/{max_rounds} · local search {searches}/{searches_per_round}')
            rows=local_transect(start,end,axis,lookup)
            def process(guess,bounds):
                nonlocal added
                if native.event_covered(guess,dataset['curves'],axes):
                    if native._session:native._session.stats['covered_seeds']+=1
                    return None
                event=native.correct_event(base,axes,guess,guess['kind'],guess['invader'],guess['solved_axis'],bounds,prepared=True)
                if add_event(event,iteration):added+=1;return event
                return None
            events,notes=native.locate_events(base,axes,rows,axis,checkpoint,max_attempts=24,on_prepared=process)
            dataset['diagnostics'].extend(dict(n,refinement_round=iteration,search=searches) for n in notes)
        # Retry a bounded number of genuinely dangling ends with an axis choice
        # aligned to the local tangent. Known junctions/presence limits are skipped.
        retries=0
        from shapely.geometry import Point,LineString
        old_curves=list(dataset['curves'])
        for ci,curve in enumerate(old_curves):
            if retries>=4:break
            if len(curve)<3:continue
            for index,neighbor in ((0,2),(-1,-3)):
                if retries>=4:break
                end=curve[index];uv=np.array([end['u'],end['v']])
                if min(*uv,*(1-uv))<JOIN_TOL:continue
                if any(LineString([(r['u'],r['v']) for r in other]).distance(Point(*uv))<JOIN_TOL
                       for j,other in enumerate(old_curves) if j!=ci and len(other)>1):continue
                if float(end.get('max_real',0.))>1e-6:continue
                if min((end[c] for c in end['community'] if c in 'MSP'),default=float('inf'))<1e-6:continue
                key=(end['kind'],end['community'],end['invader'],*np.round(uv,6))
                if key in retried_ends:continue
                retried_ends.add(key);retries+=1
                tangent=uv-np.array([curve[neighbor]['u'],curve[neighbor]['v']])
                first_axis=1-int(np.argmax(abs(tangent)))
                # Try the alternate solved control if the first chart is singular.
                for solved_axis in (first_axis,1-first_axis):
                    seed=dict(curve[neighbor],solved_axis=solved_axis)
                    try:
                        event=native.correct_event(base,axes,seed,seed['kind'],seed['invader'],solved_axis)
                        if add_event(event,iteration,allow_extension=True):added+=1;break
                    except (RuntimeError,ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
                        dataset['diagnostics'].append(dict(refinement_round=iteration,reason='Endpoint retry: '+str(exc)[:200]))
        # Feed all local checks back into region classification, not only ones
        # selected by the next random region sampling pass.
        for region in dataset['regions']:
            from shapely.geometry import Point
            old={(s['u'],s['v']) for s in region['samples']}
            for uv,roots in cache.items():
                if uv not in old and region['polygon'].contains(Point(*uv)) and region['polygon'].boundary.distance(Point(*uv))>SAMPLE_MARGIN:
                    region['samples'].append(dict(u=uv[0],v=uv[1]))
        classify()
        after=sum(r['polygon'].area for r in dataset['regions'] if r['label']=='?')
        history.append(dict(round=iteration,local_searches=searches,endpoint_retries=retries,new_curves=added,
                            new_checks=len(cache)-checked_before,unresolved_area_before=before,unresolved_area_after=after))
        if on_round is not None:on_round(dataset,list(history))
        stalled=stalled+1 if not added and after>=before-1e-8 else 0
        if not searches and not retries:
            stop_reason='no_new_candidates';break
        if stalled>=2:
            stop_reason='stalled';break
    dataset['refinement']=dict(rounds=history,equilibrium_checks=len(cache),
        stop_reason='resolved' if not any(r['label']=='?' for r in dataset['regions']) else stop_reason,
        unresolved_regions=sum(r['label']=='?' for r in dataset['regions']),
        unresolved_area=sum(r['polygon'].area for r in dataset['regions'] if r['label']=='?'),
        geometry_actions=len(dataset.get('geometry_diagnostics',[])))
    dataset['scan'].extend(r for key,roots in cache.items() if key not in initial_keys for r in roots)
    return dataset
