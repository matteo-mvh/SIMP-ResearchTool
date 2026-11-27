"""Boundary-focused sampling in a constant-volume cube; independent audit per leaf."""
from dataclasses import dataclass, field
from itertools import product
import math
import numpy as np


@dataclass
class Cell:
    lo: np.ndarray
    hi: np.ndarray
    level: int = 0
    samples: list = field(default_factory=list)

    @property
    def volume(self):
        return float(np.prod(self.hi-self.lo))

    @property
    def labels(self):
        return {label for _,label in self.samples}


def to_state(q,p,upper_s):
    """Bijective interior transform; constant Jacobian in areal coordinates."""
    q=np.asarray(q,float)
    r=np.sqrt(q[...,0]); v=q[...,1]
    return np.stack((p.N_wc*r*(1-v),upper_s*q[...,2],p.N_wc*r*v/p.H),axis=-1)


def points(cell,n,rng):
    return cell.lo+(cell.hi-cell.lo)*rng.uniform(np.finfo(float).eps,1.-np.finfo(float).eps,(n,3))


def split(cell):
    mid=(cell.lo+cell.hi)/2
    children=[]
    for side in product((0,1),repeat=3):
        side=np.array(side,bool)
        child=Cell(np.where(side,mid,cell.lo),np.where(side,cell.hi,mid),cell.level+1)
        child.samples=[(q,label) for q,label in cell.samples if np.array_equal(q>=mid,side)]
        children.append(child)
    return children


def adjacent(a,b):
    overlap=np.minimum(a.hi,b.hi)-np.maximum(a.lo,b.lo)
    return np.count_nonzero(abs(overlap)<1e-12)==1 and np.count_nonzero(overlap>1e-12)==2


def boundary_cells(cells):
    """Mixed cells and both sides of interfaces between unlike pilot labels."""
    flags=[len(c.labels)>1 or -1 in c.labels for c in cells]
    for i,a in enumerate(cells):
        for j in range(i):
            b=cells[j]
            if a.labels and b.labels and a.labels!=b.labels and adjacent(a,b):
                flags[i]=flags[j]=True
    return flags


def adaptive_samples(p,upper_s,n,budget,depth,seed,evaluate):
    """Pilot defines strata, then independent uniform audit gives unbiased volumes.

    evaluate(states,stage) returns one classification row per state. The pilot
    is not counted in the final volume estimate, preventing selection bias.
    """
    pilot_rng,audit_rng=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    root=Cell(np.zeros(3),np.ones(3)); cells=[root]; used=0
    pilots=[]

    def fill(group):
        nonlocal used
        todo=[(c,q) for c in group for q in points(c,max(0,2-len(c.samples)),pilot_rng)]
        if not todo:return
        rows=evaluate(to_state([q for _,q in todo],p,upper_s),'Boundary pilot')
        for (c,q),row in zip(todo,rows):
            label=int(row['label']);c.samples.append((q,label));pilots.append((q,label))
        used+=len(todo)

    fill(cells)
    # Allocate at least two independent audit samples per final stratum.
    while used+16<=budget and len(cells)+7<=n//2:
        flags=boundary_cells(cells)
        candidates=[(c.volume, len(c.labels),-c.level,i) for i,c in enumerate(cells)
                    if c.level<depth and (flags[i] or c.level==0)]
        if not candidates:break
        i=max(candidates)[-1]; children=split(cells[i]);fill(children)
        cells[i:i+1]=children
    flags=boundary_cells(cells)
    # Independent holdout; equal volume weights within each leaf. Extra
    # samples are preferentially assigned to possible boundary cells.
    allocation=np.full(len(cells),2,dtype=int)
    mass=np.array([c.volume*(3. if b else 1.) for c,b in zip(cells,flags)])
    remaining=n-int(allocation.sum()); expected=remaining*mass/mass.sum()
    allocation+=np.floor(expected).astype(int)
    for i in np.argsort(-(expected-np.floor(expected)))[:n-int(allocation.sum())]:allocation[i]+=1
    audit=np.concatenate([points(c,int(k),audit_rng) for c,k in zip(cells,allocation)])
    weights=np.concatenate([np.full(k,c.volume/k) for c,k in zip(cells,allocation)])
    ids=np.repeat(np.arange(len(cells)),allocation)
    states=to_state(audit,p,upper_s); rows=evaluate(states,'Independent volume audit')
    labels=np.array([r['label'] for r in rows]); metadata=[]
    for i,(c,flag,k) in enumerate(zip(cells,flags,allocation)):
        observed=sorted(set(int(v) for v in labels[ids==i])|c.labels)
        metadata.append(dict(id=i,lo=c.lo.tolist(),hi=c.hi.tolist(),level=c.level,volume_fraction=c.volume,
                             pilot_labels=sorted(c.labels),audit_count=int(k),observed_labels=observed,
                             possible_boundary=bool(flag or len(observed)>1 or -1 in observed)))
    # Mark interfaces detected by the audit, too, without changing the strata.
    for i,a in enumerate(cells):
        for j in range(i):
            if adjacent(a,cells[j]) and metadata[i]['observed_labels']!=metadata[j]['observed_labels']:
                metadata[i]['possible_boundary']=metadata[j]['possible_boundary']=True
    info=dict(pilot_count=used,leaf_count=len(cells),max_level=max(c.level for c in cells),cells=metadata,
              pilot_states=to_state([q for q,_ in pilots],p,upper_s).tolist(),pilot_labels=[l for _,l in pilots])
    return states,rows,weights,ids,info


def weighted_fractions(labels,targets,volume,weights):
    """Conditional marginal Hoeffding bounds, valid for independent stratified audit."""
    labels=np.asarray(labels);weights=np.asarray(weights,float)
    np.testing.assert_allclose(weights.sum(),1.,atol=1e-12)
    unresolved=float(weights[labels==-1].sum())
    half=math.sqrt(.5*math.log(40.)*float(weights@weights))
    rows=[]
    for i,t in [*((i,t) for i,t in enumerate(targets) if t['stable']=='stable'),(-1,dict(id='Unresolved',kind='unresolved',color='#a6a6a6'))]:
        mask=labels==i;q=float(weights[mask].sum())
        rows.append(dict(index=i,attractor=t['id'],kind=t['kind'],color=t['color'],count=int(mask.sum()),fraction=q,
                         observed_ci_low=max(0.,q-half),observed_ci_high=min(1.,q+half),
                         possible_fraction_upper=min(1.,q+(unresolved if i!=-1 else 0.)),sampled_volume=volume*q))
    return rows
