"""MP saddle stable-manifold collocation, for a single-valued M(S,HP) separator.

Exact SIMP RHS, finite-time stable-plane boundary condition, numerical quadrature.
Only report a conditional geometric estimate after independent forward checks.
Unsupported topologies or failed checks fall back to ordinary volume sampling.
"""
from __future__ import annotations
import json
import math
import time
import warnings
import numpy as np
from scipy.integrate import solve_bvp, trapezoid
from scipy.linalg import eig
from scipy.special import roots_legendre
from model.shared_eqn import growth_expressions, jacobian


class ManifoldFailure(RuntimeError):
    pass


class Trace:
    def __init__(self, p, upper_s, saddle, horizon, tol, checkpoint):
        self.p = p
        self.scale = np.array([p.N_wc, upper_s, p.N_wc/p.H])
        self.q = np.asarray(saddle.biomass)/self.scale
        J = jacobian(np.asarray(saddle.biomass),p)*self.scale[None,:]/self.scale[:,None]
        e,left = eig(J,left=True,right=False)
        self.eigenvalues=e
        if sum(e.real>1e-8)!=1 or sum(e.real<-1e-8)!=2:
            raise ManifoldFailure('MP is not a hyperbolic index-one saddle')
        w=left[:,np.argmax(e.real)].real
        self.w=w/np.linalg.norm(w)
        if abs(self.w[0])<.5:
            raise ManifoldFailure('Local stable manifold is not well conditioned as an M graph')
        self.T=horizon;self.tol=tol;self.count=0;self.checkpoint=checkpoint
        self.mesh=np.unique(np.r_[np.linspace(0,min(10,horizon),31),np.linspace(min(10,horizon),horizon,101)])
        self.max_terminal=0.;self.max_residual=0.

    def field(self,t,z):
        # Broadcast the authoritative equations across collocation nodes.
        y=z*self.scale[:,None]
        return z*np.asarray(growth_expressions(*y,self.p,np.exp))

    def solve(self,bc,previous=None,delta=None):
        self.count+=1
        if self.count%10==0:self.checkpoint(self.count)
        x=self.mesh
        guess=np.tile(self.q[:,None],(1,len(x))) if previous is None else previous.sol(x)
        if delta is not None:guess=guess+np.asarray(delta)[:,None]*np.exp(-x/30.)
        # Newton iterations may try nonphysical intermediate guesses. Reject
        # final solutions with nonphysical paths; never clip the vector field.
        with warnings.catch_warnings(),np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            warnings.simplefilter('ignore',RuntimeWarning)
            return solve_bvp(self.field,bc,x,guess,tol=self.tol,max_nodes=3000)

    def valid(self,sol,physical=True):
        if sol is None or not sol.success:return False
        # Check at collocation nodes and their midpoints, including endpoints.
        a=sol.sol(np.unique(np.r_[sol.x,(sol.x[:-1]+sol.x[1:])/2]))
        if not np.isfinite(a).all() or a.min()<-2e-7:return False
        if physical and np.max(a[0]+a[2])>1+2e-7:return False
        end=float(np.max(abs(a[:,-1]-self.q)))
        if end>2e-3:return False
        self.max_terminal=max(self.max_terminal,end)
        self.max_residual=max(self.max_residual,float(np.max(sol.rms_residuals)))
        return True

    def fixed(self,P,S,prev):
        a=self.q if prev is None else prev.y[:,0]
        return self.solve(lambda x,y:np.array([x[2]-P,x[1]-S,self.w@(y-self.q)]),
                          prev,[0,S-a[1],P-a[2]])

    def advance(self,P,S,prev):
        # Predictor/corrector continuation with smaller steps on failures.
        start=prev.y[:,0]; distance=max(abs(P-start[2]),abs(S-start[1]))
        if distance<1e-12:return prev
        n=max(1,math.ceil(distance/.02)); i=0
        while i<n:
            f=(i+1)/n
            pp=start[2]+f*(P-start[2]);ss=start[1]+f*(S-start[1])
            r=self.fixed(pp,ss,prev)
            if self.valid(r):prev=r;i+=1;continue
            # Retry halfway from the last accepted point. Do not interpret a
            # failed nonlinear solve as evidence that the manifold has ended.
            a=prev.y[:,0];half=self.fixed((pp+a[2])/2,(ss+a[1])/2,prev)
            if not self.valid(half):return None
            r=self.fixed(pp,ss,half)
            if not self.valid(r):return None
            prev=r;i+=1
        return prev

    def base_edge(self):
        # S=0 and zero initial dissolved N. Its intersection with W^s(MP)
        # determines the end of the base graph in the feasible triangle.
        r=self.solve(lambda x,y:np.array([x[1],x[0]+x[2]-1,self.w@(y-self.q)]),
                     delta=[.01,0,1-self.q[0]-self.q[2]-.01])
        if not self.valid(r) or not self.q[2]<r.y[2,0]<1:
            raise ManifoldFailure('Could not establish physical S=0 edge intersection')
        return r

    def cap_edge(self,P,s,prev,failed):
        # M+HP=N_wc at fixed P; solve for S instead of M. Try both the
        # last physical point and a converged (slightly external) predictor.
        a=prev.y[:,0];bc=lambda x,y:np.array([x[2]-P,x[0]+x[2]-1,self.w@(y-self.q)])
        seeds=[(prev,[1-P-a[0],max(.002,.5*(s-a[1])),0])]
        if self.valid(failed,physical=False):
            b=failed.y[:,0];seeds.append((failed,[1-P-b[0],0,0]))
        # Last valid point can be S=0, where the derivative is small; a
        # positive S predictor prevents a singular free-S boundary condition.
        seeds.extend((prev,[1-P-a[0],max(.001,k*(s-a[1])),0]) for k in (.25,.75,1.))
        for guess,delta in seeds:
            r=self.solve(bc,guess,delta)
            if self.valid(r) and a[1]-1e-7<=r.y[1,0]<=s+1e-6:
                return r
        # Reparameterize the curve by initial macroalgal biomass near the
        # nitrogen face. Continue in M rather than jumping to the cap.
        good=prev
        for _ in range(40):
            aa=good.y[:,0]; gap=1-P-aa[0]
            if gap<=2e-7:
                r=self.solve(bc,good,[gap,0,0])
                if self.valid(r) and a[1]-1e-7<=r.y[1,0]<=s+1e-6:return r
                break
            dm=min(gap,.002)
            accepted=False
            for _retry in range(8):
                mm=aa[0]+dm
                rr=self.solve(lambda x,y:np.array([x[2]-P,x[0]-mm,self.w@(y-self.q)]),good,[dm,0,0])
                if self.valid(rr) and aa[1]-1e-7<=rr.y[1,0]<=s+1e-6:
                    good=rr;accepted=True;break
                dm*=.5
            if not accepted:break
        raise ManifoldFailure(f'Unresolved surface intersection P={P:.8g}, S={s:.8g}')

    def calculate(self,n,s_step):
        edge=self.base_edge(); pe=float(edge.y[2,0]);x,w=roots_legendre(n)
        coords=(x+1)*pe/2;weights=w*pe/2
        base=edge;volume=0.;rows=[]
        for P,weight in zip(coords[::-1],weights[::-1]):
            P=float(P);base=self.advance(P,0.,base)
            if base is None:raise ManifoldFailure(f'Base graph continuation failed at P={P:.8g}')
            path=[base.y[:,0].copy()];prev=base
            for S in np.linspace(0,1,math.ceil(1/s_step)+1)[1:]:
                S=float(S);r=self.fixed(P,S,prev)
                if self.valid(r):
                    path.append(r.y[:,0].copy());prev=r;continue
                # If subdivision resolves a numerical issue, continue rather
                # than mistakenly declaring a physical boundary intersection.
                mid=(prev.y[1,0]+S)/2;half=self.fixed(P,mid,prev)
                if self.valid(half):
                    retry=self.fixed(P,S,half)
                    if self.valid(retry):
                        path.extend([half.y[:,0].copy(),retry.y[:,0].copy()]);prev=retry;continue
                    path.append(half.y[:,0].copy());prev=half;r=retry
                cut=self.cap_edge(P,S,prev,r);path.append(cut.y[:,0].copy());break
            path=np.asarray(path);thickness=1-P-path[:,0]
            if np.min(thickness)<-3e-7 or np.min(np.diff(path[:,1]))<-1e-9:
                raise ManifoldFailure('Nonphysical or nonordered graph')
            # This implementation specifically tests a nondecreasing M(S)
            # graph. Nonmonotone topology is unsupported, not forced to fit.
            if np.min(np.diff(path[:,0])) < -2e-5:
                raise ManifoldFailure('Nonmonotone M(S) graph: use volume sampling')
            thickness=np.maximum(0,thickness)
            volume+=2*float(weight)*float(trapezoid(thickness,path[:,1]))
            rows.append(dict(P=P,points=path.tolist(),thickness=thickness.tolist()))
        return dict(upper_fraction=float(volume),P_edge=pe,rows=rows[::-1],solves=self.count,
                    max_terminal=self.max_terminal,max_residual=self.max_residual,
                    horizon=self.T,tolerance=self.tol,quadrature_nodes=n,S_step=s_step)


def graph_thickness(trace,P,S):
    """Interpolation for visualization and audit only; quadrature uses traced rows."""
    rows=trace['rows']; pe=trace['P_edge']
    if P>=pe:return 0.
    ps=np.array([r['P'] for r in rows]+[pe]);i=int(np.searchsorted(ps,P))
    def val(k):
        if k>=len(rows):return 0.
        r=rows[k];a=np.asarray(r['points']);d=np.asarray(r['thickness'])
        return float(np.interp(S,a[:,1],d,right=0.))
    if i==0:return val(0)
    w=(P-ps[i-1])/(ps[i]-ps[i-1]);return (1-w)*val(i-1)+w*val(i)


def audit(trace,p,upper_s,targets,states,evaluate):
    scale=np.array([p.N_wc,upper_s,p.N_wc/p.H]); candidates=[]
    rows=trace['rows']
    # Independent paired forward integrations on either side of actual
    # collocation points, away from the sampling-domain faces.
    for k in np.linspace(0,len(rows)-1,min(8,len(rows)),dtype=int):
        a=np.asarray(rows[k]['points']);P=rows[k]['P']
        viable=a[(a[:,0]>.008)&(a[:,0]<1-P-.008)&(a[:,1]>.005)&(a[:,1]<.995)]
        if len(viable):
            q=viable[len(viable)//2];d=min(.004,q[0]/3,(1-P-q[0])/3)
            candidates.extend([q-np.array([d,0,0]),q+np.array([d,0,0])])
    if len(candidates)<6:raise ManifoldFailure('Too few interior two-sided surface checks')
    result=evaluate(np.array(candidates)*scale,'Manifold side checks')
    labels=np.array([r['label'] for r in result]);lo=set(labels[::2]);hi=set(labels[1::2])
    stable={i for i,t in enumerate(targets) if t['stable']=='stable'}
    if len(lo)!=1 or len(hi)!=1 or lo|hi!=stable or lo==hi:
        raise ManifoldFailure(f'Forward checks do not establish a consistent separator: {labels.tolist()}')
    lower=int(labels[0]);upper=int(labels[1]);q=states/scale
    predicted=np.array([upper if m>1-P-graph_thickness(trace,P,S) else lower for m,S,P in q])
    result=evaluate(states,'Independent uniform manifold audit');actual=np.array([r['label'] for r in result])
    return dict(lower=lower,upper=upper,side_count=len(candidates),side_labels=labels.tolist(),
                predicted=predicted,actual=actual,result=result)


def save_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def run_manifold(p,upper_s,eq,targets,discovery,values,ctx,*,target_seconds=0.):
    """Return True after a successful conditional geometry result, else fall back."""
    from scripts.basin_runner import sample_initial,fractions,export_rows
    from scripts.basin_fast import BasinPool
    started=time.monotonic();folder=ctx.folder;report=dict(status='running')
    def fail(reason):
        report.update(status='fallback',reason=str(reason),seconds=time.monotonic()-started)
        save_json(folder/'basin_manifold.json',report)
        print('Manifold unsupported/unvalidated; using adaptive volume sampling:',reason,flush=True)
        return False
    stable=[t for t in targets if t['stable']=='stable']
    if any(e.stability.classification=='nonhyperbolic' for e in eq):
        return fail('Nonhyperbolic equilibrium: use general sampling near bifurcations')
    if len(stable)!=2 or any(t['kind']!='equilibrium' for t in stable):
        return fail('Requires exactly two detected stable equilibria and no verified stable cycle')
    if discovery.get('status') not in ('not_confirmed','analytically_excluded_MSP'):
        return fail('Cycle discovery must be enabled and completed without a detected cycle')
    # Specialized, deliberately restricted first implementation: MP is on
    # S=0 and its stable subspace must enter S>0. Reject competing interior
    # index-one saddles; no general topology assumption is silently made.
    candidates=[]
    for e in eq:
        y=np.asarray(e.biomass);J=jacobian(y,p);ev=np.linalg.eigvals(J)
        if sum(ev.real>1e-8)==1 and sum(ev.real<-1e-8)==2:
            absent=y<=1e-8
            if np.any(np.diag(J)[absent]>1e-8):continue
            candidates.append(e)
    if len(candidates)!=1 or candidates[0].community!='MP':
        return fail('Requires a unique index-one saddle with interior stable manifold, on the MP face')
    saddle=candidates[0];ev=np.asarray(saddle.stability.eigenvalues)
    T=max(300.,10/min(-ev.real[ev.real<0]));days=float(values.get('basin_days',20000.))
    if T*1.1>days:return fail('Integration horizon too short for terminal approach to the saddle')
    n=int(values.get('basin_manifold_resolution',12));audit_n=int(values.get('basin_manifold_audit',64))
    report.update(saddle='MP',saddle_state=list(saddle.biomass),eigenvalues=[[float(v.real),float(v.imag)] for v in ev],
                  method='forward-time stable-manifold BVP; M graph over S, HP; finite terminal plane')
    try:
        def progress(stage,base):
            return lambda count:ctx.checkpoint(base,f'Manifold {stage} · {count} BVP solves')
        ctx.checkpoint(5,'Tracing coarse stable manifold',force=True)
        a=Trace(p,upper_s,saddle,T,1e-6,progress('coarse',10)).calculate(n,.04)
        report['coarse']={k:v for k,v in a.items() if k!='rows'}
        ctx.checkpoint(35,'Refining stable manifold and terminal horizon',force=True)
        b=Trace(p,upper_s,saddle,T*1.1,1e-6,progress('refined',40)).calculate(2*n,.025)
        report['refined']={k:v for k,v in b.items() if k!='rows'}
        if not all(0<=r['upper_fraction']<=1 for r in (a,b)):
            return fail('Nonphysical geometric volume')
        delta=abs(a['upper_fraction']-b['upper_fraction']);report['refinement_difference']=delta
        if delta>.003:return fail('Volume changed by >0.3 percentage points on refinement; increase resolution')
        states=sample_initial(p,upper_s,audit_n,int(values.get('basin_seed',42))+117)
        workers=min(int(values.get('basin_workers',4)),audit_n)
        # Reference backend avoids Numba startup and audits the authoritative RHS.
        def check(done,stage,seconds):ctx.checkpoint(65+20*min(1,done/(audit_n+16)),f'{stage} · {done} integrations')
        with BasinPool(p,targets,days,workers,'reference',check) as pool:
            check_result=audit(b,p,upper_s,targets,states,pool.evaluate)
        lower,upper=check_result['lower'],check_result['upper'];actual=check_result['actual'];pred=check_result['predicted']
        vol=.5*p.N_wc**2*upper_s;ar=fractions(actual,targets,vol)
        unresolved=int(np.sum(actual<0));disagree=int(np.sum(actual!=pred))
        q=b['upper_fraction'];upper_audit=next(r for r in ar if r['index']==upper)
        report.update(audit_count=audit_n,audit_unresolved=unresolved,audit_disagreements=disagree,
                      side_count=check_result['side_count'],side_labels=check_result['side_labels'],
                      upper_target=targets[upper]['id'],lower_target=targets[lower]['id'],audit_fractions=ar)
        export_rows(folder/'basin_manifold_audit.csv',[dict(M0=float(y[0]),S0=float(y[1]),P0_area=float(y[2]*p.H),
             predicted=int(pr),reference=int(ac),agree=bool(pr==ac)) for y,pr,ac in zip(states,pred,actual)])
        # A resolved mismatch indicates interpolation/geometry error or a
        # missed component. Do not present a surface-only estimate as verified.
        if unresolved or disagree or not upper_audit['observed_ci_low']<=q<=upper_audit['observed_ci_high']:
            return fail('Independent audit did not fully agree with reconstructed graph; see audit CSV')
        report.update(status='conditional_geometry_passed',upper_fraction=q,lower_fraction=1-q,
                      seconds=time.monotonic()-started,limitations='Assumes one single-valued separator connected to MP; finite forward audit cannot exclude additional components or cycles. Refinement difference is not a rigorous error bound.')
        save_json(folder/'basin_manifold.json',report);save_json(folder/'basin_manifold_surface.json',b)
        estimates=[dict(index=i,attractor=targets[i]['id'],kind='equilibrium',color=targets[i]['color'],
                        fraction=f,sampled_volume=f*vol,method='conditional manifold quadrature',
                        refinement_difference=delta,audit_fraction=next(r['fraction'] for r in ar if r['index']==i),
                        audit_ci_low=next(r['observed_ci_low'] for r in ar if r['index']==i),
                        audit_ci_high=next(r['observed_ci_high'] for r in ar if r['index']==i))
                   for i,f in [(upper,q),(lower,1-q)]]
        export_rows(folder/'basin_fractions.csv',estimates)
        export_rows(folder/'basin_samples.csv',[dict(index=i,M0=float(y[0]),S0=float(y[1]),P0_area=float(y[2]*p.H),
                    attractor_index=int(ac),attractor=targets[ac]['id'] if ac>=0 else 'Unresolved',
                    days=check_result['result'][i]['days'],reason=check_result['result'][i]['reason']) for i,(y,ac) in enumerate(zip(states,actual))])
        ctx.checkpoint(92,'Plotting manifold geometry',force=True)
        plot_geometry(folder/'figures',p,upper_s,targets,lower,upper,b,a,ar)
        save_json(folder/'summary.json',dict(seconds=time.monotonic()-started+target_seconds,target_discovery_seconds=target_seconds,N_wc=p.N_wc,H=p.H,S_max=upper_s,
                  sampling_mode='manifold',requested_sampling_mode='hybrid',samples=audit_n,domain_volume_areal=vol,fractions=estimates,**{k:v for k,v in report.items() if k!='seconds'}))
        ctx.checkpoint(100,'Completed basin manifold analysis',force=True)
        return True
    except (ManifoldFailure,ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
        return fail(exc)


def plot_geometry(folder,p,upper_s,targets,lower,upper,b,a,audit_rows):
    from scripts.continuation_plots import plt,save
    from matplotlib.ticker import PercentFormatter
    folder.mkdir(exist_ok=True)
    q=b['upper_fraction'];fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained')
    labels=[targets[i]['id'] for i in (upper,lower)];cols=[targets[i]['color'] for i in (upper,lower)]
    ax.barh(labels,[q,1-q],color=cols,alpha=.85)
    for y,f in enumerate((q,1-q)):ax.text(min(.99,f+.015),y,f'{f:.2%}',va='center',ha='right' if f>.9 else 'left')
    ax.set(xlim=(0,1.04),xlabel='Fraction of sampled domain volume',title='Stable-manifold volume estimate (conditional)');ax.xaxis.set_major_formatter(PercentFormatter(1));ax.invert_yaxis()
    fig.supxlabel(f'Coarse/refined difference: {abs(q-a["upper_fraction"])*100:.3f} percentage points; independent forward checks passed.',fontsize=8)
    save(fig,folder,'basin_volume_fractions')
    # Plot the actual traced row curves; do not invent a filled surface
    # beyond the observed physical edge.
    fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(111,projection='3d')
    for row in b['rows']:
        pts=np.asarray(row['points'])*np.array([p.N_wc,upper_s,p.N_wc]);ax.plot(*pts.T,color='#555555',lw=1)
    ax.set(xlabel='M₀',ylabel='S₀',zlabel='HP₀',title='Traced stable-manifold curves (mmol N/m²)',xlim=(0,p.N_wc),ylim=(0,upper_s),zlim=(0,p.N_wc))
    fig.tight_layout();save(fig,folder,'basin_manifold_3d')
    fig,axes=plt.subplots(1,3,figsize=(10,3.7),sharey=True,layout='constrained')
    for ax,P in zip(axes,(.05,.2,.4)):
        S=np.linspace(0,1,201);th=np.array([graph_thickness(b,P,s) for s in S]);M=(1-P-th)*p.N_wc
        ax.fill_betweenx(S*upper_s,0,M,color=cols[1],alpha=.7);ax.fill_betweenx(S*upper_s,M,(1-P)*p.N_wc,color=cols[0],alpha=.7)
        ax.plot(M,S*upper_s,color='k',lw=1);ax.set(xlim=(0,p.N_wc),xlabel='M₀',title=f'HP₀ = {P*p.N_wc:g}')
    axes[0].set_ylabel('S₀');fig.suptitle('Interpolated basin sections (mmol N/m²)');save(fig,folder,'basin_manifold_sections')
