"""Conservative topology and stable-equilibrium region classification.

Native curve records are never changed here. Geometry-only joins are bounded
by a small normalized tolerance and are reported separately. A region with
conflicting interior evidence is always unresolved, not majority-coloured.
"""
import numpy as np
from shapely.geometry import LineString, box, Point
from shapely.ops import unary_union, polygonize, snap
from shapely import set_precision
from model.pyoomph_continuation import discover, covered

# Both tolerances are in normalized parameter coordinates, not physical units.
JOIN_TOL=2e-5
SAMPLE_MARGIN=4e-5
MIN_POLYGON_AREA=1e-9


def family(curve):
    return tuple(curve[0].get(k) for k in ('community','kind','invader'))


def curve_network(curves):
    domain=box(0,0,1,1)
    items=[];notes=[]
    for curve in curves:
        if len(curve)<2:continue
        coords=np.array([(r['u'],r['v']) for r in curve],float)
        if not np.all(np.isfinite(coords)):continue
        # Snap endpoints only; do not move interior native samples.
        for idx in (0,-1):
            for axis in (0,1):
                if abs(coords[idx,axis])<=JOIN_TOL:coords[idx,axis]=0.
                if abs(coords[idx,axis]-1)<=JOIN_TOL:coords[idx,axis]=1.
        line=LineString(coords).intersection(domain)
        if line.is_empty or line.length==0:continue
        # Curves are recorded in order, and ordinarily wholly in the domain.
        if line.geom_type!='LineString':
            for part in getattr(line,'geoms',[]):
                if part.geom_type=='LineString' and part.length>0:items.append([part,curve])
        else:items.append([line,curve])
    items.sort(key=lambda pair:pair[0].length,reverse=True)
    accepted=[]
    for line,curve in items:
        # Suppress retraces only when both parameter-space and resident state
        # agree. Distinct nearby branches must not be collapsed.
        duplicate=False
        for old,old_curve in accepted:
            if family(curve)!=family(old_curve):continue
            if old.buffer(JOIN_TOL).covers(line) and all(
                    covered(r,[old_curve],(None,None),tolerance=1e-4)
                    for r in curve[::max(1,len(curve)//20)]+[curve[-1]]):
                duplicate=True;break
        if duplicate:notes.append(dict(action='duplicate_curve_geometry',family=family(curve)));continue
        accepted.append([line,curve])
    # Node close endpoints explicitly. Merely checking distance to another
    # curve does NOT connect a line to it for polygonization.
    lines=[pair[0] for pair in accepted]
    for i in range(len(lines)):
        for end_index in (0,-1):
            line=lines[i];coords=list(line.coords);end=Point(coords[end_index])
            if domain.boundary.distance(end)<=1e-9:continue
            if Point(coords[0]).distance(Point(coords[-1]))<=JOIN_TOL:
                coords[-1]=coords[0];lines[i]=LineString(coords)
                notes.append(dict(action='close_loop',family=family(accepted[i][1])));break
            options=[(other.distance(end),j,other) for j,other in enumerate(lines) if j!=i]
            if not options:continue
            dist,j,other=min(options,key=lambda option:option[0])
            if dist>JOIN_TOL:continue
            target=other.interpolate(other.project(end))
            # Insert an identical node into both geometries. Keep the source
            # endpoint itself and append a short connection, rather than
            # moving an interior verified curve point.
            xy=(target.x,target.y)
            if dist>1e-12:
                if end_index==0:coords.insert(0,xy)
                else:coords.append(xy)
                lines[i]=LineString(coords)
            lines[j]=snap(other,target,1e-10)
            if dist>1e-12:notes.append(dict(action='endpoint_join',distance=float(dist),
                         family=family(accepted[i][1]),target_family=family(accepted[j][1])))
    network=unary_union([set_precision(line,1e-9) for line in [domain.boundary,*lines]])
    all_polygons=[p for p in polygonize(network) if domain.covers(p.representative_point())]
    polygons=[p for p in all_polygons if p.area>MIN_POLYGON_AREA]
    discarded=[p for p in all_polygons if p.area<=MIN_POLYGON_AREA]
    if discarded:notes.append(dict(action='subresolution_polygons',count=len(discarded),
                                   area=sum(p.area for p in discarded)))
    if not polygons:polygons=[domain]
    # Only endpoints of a potentially stability-changing event should grey
    # an otherwise consistent stable-state polygon. Unstable-only resident
    # events are still exported as curves and diagnostics.
    incomplete=[]
    for i,line in enumerate(lines):
        coords=list(line.coords)
        if Point(coords[0]).distance(Point(coords[-1]))<=1e-9:continue
        for idx in (0,-1):
            end=Point(coords[idx]);row=accepted[i][1][idx]
            if domain.boundary.distance(end)<=1e-8:continue
            if any(other.distance(end)<=1e-8 for j,other in enumerate(lines) if j!=i):continue
            if float(row.get('max_real',0.))<=1e-6:incomplete.append(end)
            notes.append(dict(action='dangling_endpoint',family=family(accepted[i][1]),
                              u=end.x,v=end.y,stability_relevant=float(row.get('max_real',0.))<=1e-6))
    return polygons,incomplete,notes


def signature(roots):
    stable=[r for r in roots if r['stability']=='stable']
    # Nonhyperbolicity matters if the equilibrium itself may be stable.
    # An unstable root with a separate neutral eigenmode does not change
    # the stable-equilibrium set. classify_stability already makes this distinction.
    if not roots or any(r['stability']=='nonhyperbolic' for r in roots):return '?',stable
    return '+'.join(sorted(r['community'] for r in stable)) or '∅',stable


def analyze_regions(dataset,base,axes,samples,checkpoint,lookup=None):
    polygons,incomplete,notes=curve_network(dataset['curves'])
    rng=np.random.default_rng(1042)
    # Include all earlier interior witnesses, including initial transect
    # samples, so random resampling cannot accidentally hide conflicts.
    previous={(float(s['u']),float(s['v'])) for r in dataset.get('regions',[]) for s in r['samples']}
    previous.update((float(s['u']),float(s['v'])) for s in dataset.get('scan',[]))
    regions=[]
    for i,poly in enumerate(polygons):
        point=poly.representative_point()
        coords=[(point.x,point.y)]
        coords.extend(uv for uv in sorted(previous) if poly.contains(Point(*uv))
                      and poly.boundary.distance(Point(*uv))>SAMPLE_MARGIN)
        coords=list(dict.fromkeys(coords))
        # Prefer samples well inside polygons; narrow polygons receive a
        # smaller proportional margin rather than no independent samples.
        margin=min(SAMPLE_MARGIN,poly.boundary.distance(point)*.25)
        lo_x,lo_y,hi_x,hi_y=poly.bounds
        target=max(len(coords),max(12,samples*3))
        # Even scan-rich polygons need fresh independent interior witnesses.
        target=max(target,len(coords)+min(samples,5))
        for _ in range(samples*500):
            if len(coords)>=target:break
            uv=(rng.uniform(lo_x,hi_x),rng.uniform(lo_y,hi_y));pt=Point(*uv)
            if poly.contains(pt) and poly.boundary.distance(pt)>margin:coords.append(uv)
        records=[];signatures=[]
        for j,uv in enumerate(coords):
            checkpoint(f'Checking region {i+1}/{len(polygons)} · sample {j+1}/{len(coords)}')
            roots=lookup(uv) if lookup is not None else discover(base,axes,uv)
            sig,stable=signature(roots);signatures.append(sig)
            selected=max(stable,key=lambda r:(r['S'],r['M'],r['P'])) if stable else None
            records.append(dict(u=uv[0],v=uv[1],selected=selected,roots=roots,signature=sig))
        label=signatures[0] if len(set(signatures))==1 else '?'
        dangling=any(poly.distance(end)<JOIN_TOL for end in incomplete)
        reasons=[]
        if len(set(signatures))>1:reasons.append('conflicting_samples')
        if '?' in signatures:reasons.append('marginal_equilibrium')
        if dangling:label='?';reasons.append('incomplete_boundary')
        regions.append(dict(polygon=poly,label=label,samples=records,signatures=signatures,reasons=reasons))
    dataset['regions']=regions;dataset['geometry_diagnostics']=notes
    return dataset
