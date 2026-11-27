"""Pure numerical helpers for deterministic periodic guesses and certification."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from model.shared_eqn import jacobian, per_capita, growth_expressions


def admissible_msp(row,p,tolerance=1e-7):
    if row.get('community')!='MSP':return False
    y=np.array([row[c] for c in 'MSP'],float)
    return bool(np.all(np.isfinite(y)) and np.min(y)>1e-8 and
                p.N_wc-y[0]-p.H*y[2]>=-tolerance and np.max(abs(per_capita(y,p)))<=tolerance)


def absence_certificate(p, intervals=1024):
    """Conservative interval bounds on the reduced MSP scalar residual.

    Returns a sufficient exclusion certificate, never infers absence from a
    finite root search. Bounds include every M,S,P satisfying F_P=F_S=0.
    Floating bounds are padded; near-degenerate cases return inconclusive.
    """
    if min(p.k_S,p.m_S,p.m_M,p.m_P+p.g_P)<=0:return None
    light=p.I_0*np.exp(-p.H*p.kw/2)
    loss=p.m_P+p.g_P
    margin=1e-10
    if p.mu_P*light/(p.K_I_P+light)-loss < -margin:
        return 'Maximum possible phytoplankton growth is below its loss rate'
    lightS=p.I_0*np.exp(-p.H*p.kw)
    if p.mu_S*lightS/(p.K_I_S+lightS)-p.m_S < -margin:
        return 'Maximum possible seagrass growth is below its loss rate'
    if p.N_wc<=0:return 'Zero available nitrogen inventory'
    # Cover the entire P inventory interval, not only brackets found by discovery.
    edges=np.linspace(0,p.N_wc/p.H,intervals+1)
    plo,phi=edges[:-1],edges[1:]
    iplo=p.I_0*np.exp(-p.H*(p.kw+p.k_P*phi)/2)
    iphi=p.I_0*np.exp(-p.H*(p.kw+p.k_P*plo)/2)
    dlo=p.mu_P*iplo/(p.K_I_P+iplo)-loss
    dhi=p.mu_P*iphi/(p.K_I_P+iphi)-loss
    feasible=dhi>0
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        nlo=p.K_N_P*loss/dhi
        nhi=np.where(dlo>0,p.K_N_P*loss/dlo,np.inf)
        mlo=np.maximum(0,p.N_wc-p.H*(phi+nhi))
        mhi=p.N_wc-p.H*(plo+nlo)
        feasible &= mhi>0
        mhi=np.maximum(mhi,0)
        slo_loss=p.m_S+p.m_SM*mlo
        shi_loss=p.m_S+p.m_SM*mhi
        feasible &= slo_loss<p.mu_S
        reqlo=p.K_I_S*slo_loss/(p.mu_S-slo_loss)
        reqhi=np.where(shi_loss<p.mu_S,p.K_I_S*shi_loss/(p.mu_S-shi_loss),np.inf)
        slo=np.maximum(0,(np.log(p.I_0/reqhi)-p.H*(p.kw+p.k_P*phi)-p.k_M*mhi)/p.k_S)
        shi=(np.log(p.I_0/reqlo)-p.H*(p.kw+p.k_P*plo)-p.k_M*mlo)/p.k_S
        feasible &= shi>0
        shi=np.maximum(shi,0)
        imlo=p.I_0*np.exp(-p.H*(p.kw+p.k_P*phi)-p.k_M*mhi/2)
        imhi=p.I_0*np.exp(-p.H*(p.kw+p.k_P*plo)-p.k_M*mlo/2)
        from model.shared_eqn import grazing
        flo=p.mu_M*imlo/(p.K_I_M+imlo)*nlo/(p.K_N_M+nlo)-p.m_M-grazing(shi,p)
        nfrac=np.where(np.isinf(nhi),1,nhi/(p.K_N_M+nhi))
        fhi=p.mu_M*imhi/(p.K_I_M+imhi)*nfrac-p.m_M-grazing(slo,p)
    ruled_out=(flo>margin)|(fhi < -margin)
    if np.all(~feasible | ruled_out):return f'MSP reduced residual excluded on {intervals} covering intervals'
    return None


def deterministic_guesses(row,p,options):
    """Log-state ellipses from complex modes, or pairs of real eigenvectors."""
    y=np.array([row[c] for c in 'MSP'],float)
    eigenvalues,vectors=np.linalg.eig(jacobian(y,p))
    modes=[]
    for i,value in enumerate(eigenvalues):
        if value.imag>1e-8:
            z=vectors[:,i]/y
            modes.append((z.real,-z.imag,2*np.pi/abs(value.imag),f'complex_{i}'))
    order=np.argsort(abs(eigenvalues))
    planes=((0,1),(0,2),(1,2)) if not modes or options['periodic_real_modes']=='always' else ()
    for a,b in planes:
        i,j=order[a],order[b]
        # Real parts alone may be collinear for a complex pair. Coordinate
        # directions provide a deterministic fallback for that degeneracy.
        first=vectors[:,i].real/y;second=vectors[:,j].real/y
        if np.linalg.matrix_rank(np.stack([first,second]))<2:second=np.eye(3)[b]
        rate=max(min(abs(eigenvalues[i]),abs(eigenvalues[j])),1e-6)
        modes.append((first,second,2*np.pi/rate,f'real_plane_{i}_{j}'))
    for a,b,period,mode in modes:
        norm=max(np.max(np.sqrt(a*a+b*b)),1e-15);a=a/norm;b=b/norm
        for amplitude in options['periodic_amplitudes']:
            for factor in options['periodic_period_factors']:
                T=float(np.clip(period*factor,options['periodic_min_period'],options['periodic_max_period']))
                yield dict(kind='manual',equilibrium=row,period=T,a=a.tolist(),b=b.tolist(),
                           amplitude=amplitude,mode=mode)


def floquet_classification(multipliers,tolerance):
    values=np.asarray(multipliers,complex)
    out=dict(floquet=[[float(v.real),float(v.imag)] for v in values],stability='unresolved',trivial_index=None)
    if len(values)!=3 or not np.all(np.isfinite(values)):return out
    index=int(np.argmin(abs(values-1)));error=float(abs(values[index]-1))
    out.update(trivial_index=index,trivial_error=error)
    if error>tolerance:return out
    nontrivial=np.delete(values,index)  # Remove exactly ONE, even at a cycle fold.
    radius=float(np.max(abs(nontrivial)));out['spectral_radius']=radius
    out['stability']='unstable' if radius>1+tolerance else 'stable' if radius<1-tolerance else 'nonhyperbolic'
    return out


def spectrum_distance(a,b):
    a=np.asarray([complex(*v) for v in a]);b=np.asarray([complex(*v) for v in b])
    if len(a)!=len(b):return float('inf')
    i,j=linear_sum_assignment(abs(a[:,None]-b[None,:]))
    return float(np.max(abs(a[i]-b[j])/np.maximum(1,abs(b[j]))))


def orbit_signature(row):
    return np.array([np.log(row['period']),*[np.log1p(row[c+'_'+bound]) for c in ('M','S','P_area')
                                           for bound in ('min','max','mean')]])


def same_orbit(a,b,tolerance=.01):
    return bool(np.max(abs(orbit_signature(a)-orbit_signature(b)))<tolerance)


def sample_metrics(states,period,p,options):
    """Check dense periodic samples, including an independent ODE defect."""
    y=np.asarray(states,float)
    if y.ndim!=2 or y.shape[1]!=3 or not np.all(np.isfinite(y)):
        raise ValueError('Nonfinite orbit samples')
    if not options['periodic_min_period']<=period<=options['periodic_max_period']:
        raise ValueError('Period outside configured limits')
    if np.min(y)<=1e-8:raise ValueError('Orbit is not strictly MSP-positive')
    n=p.N_wc-y[:,0]-p.H*y[:,2]
    if np.min(n)<-1e-8*max(1,p.N_wc):raise ValueError('Negative dissolved nitrogen')
    amplitude=float(np.max(np.ptp(y,axis=0)/np.maximum(np.mean(y,axis=0),1e-8)))
    if amplitude<options['periodic_min_amplitude']:raise ValueError('Collapsed stationary solution')
    # Periodic centered derivative: independent of the native collocation residual.
    dt=period/len(y)
    derivative=(-np.roll(y,-2,axis=0)+8*np.roll(y,-1,axis=0)-8*np.roll(y,1,axis=0)+np.roll(y,2,axis=0))/(12*dt)
    rhs=y*np.asarray(growth_expressions(*y.T,p,np.exp)).T
    denominator=np.maximum(np.max(abs(rhs),axis=0),np.mean(y,axis=0)/period*1e-3)
    defect=float(np.max(np.max(abs(derivative-rhs),axis=0)/denominator))
    # Some pyoomph samplers interpolate linearly between collocation nodes;
    # differentiating that output gives knot artifacts. Also check the ODE's
    # integral balance by quadrature of the already-computed periodic curve.
    # This does not solve an IVP or advance a state in time.
    increments=dt*(rhs+np.roll(rhs,-1,axis=0))/2
    integral=np.vstack((np.zeros(3),np.cumsum(increments,axis=0)))
    closed=np.vstack((y,y[0]))
    balance=float(np.max(abs(closed-y[0]-integral)/np.maximum(np.ptp(y,axis=0),np.mean(y,axis=0)*1e-4)))
    area=np.column_stack((y[:,0],y[:,1],p.H*y[:,2],n))
    return dict(period=float(period),relative_amplitude=amplitude,differential_defect=defect,balance_defect=balance,
                nitrogen_min=float(n.min()),inventory_error=float(np.max(abs(area[:,0]+area[:,2]+area[:,3]-p.N_wc))),
                **{c+'_'+name:float(fn(area[:,i])) for i,c in enumerate(('M','S','P_area','N_area'))
                   for name,fn in (('min',np.min),('max',np.max),('mean',np.mean))})
