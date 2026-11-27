"""Fast exact SIMP log-coordinate kernels; SciPy LSODA remains the integrator."""
from __future__ import annotations
import math
import numpy as np
from scipy.integrate import solve_ivp
from model.shared_eqn import per_capita, jacobian
from scripts.phase_space import init_worker, classify_seed

KEYS = ('H','N_wc','I_0','kw','k_P','k_M','k_S','mu_M','mu_S','mu_P',
        'K_I_M','K_I_S','K_I_P','K_N_M','K_N_P','m_M','m_S','m_P','g_P',
        'g_M','K_G_S','gamma','beta')


def coefficients(p):
    return np.array([getattr(p,k) for k in KEYS]+[p.m_SM], dtype=float)


def scalar_rhs(z, a):
    H,Nwc,I0,kw,kp,km,ks,um,us,up,kim,kis,kip,knm,knp,mm,ms,mp,gp,gm,kg,gamma,beta,msm=a
    M,S,P=math.exp(z[0]),math.exp(z[1]),math.exp(z[2])
    att=H*(kw+kp*P)
    IM=I0*math.exp(-att-km*M/2)
    IS=I0*math.exp(-att-km*M-ks*S)
    IP=I0*math.exp(-att/2)
    N=(Nwc-M)/H-P
    x=S/kg; xb=x**beta
    g=gm*(1+gamma)*xb/(2*(1+gamma*xb))
    return np.array((um*IM/(kim+IM)*N/(knm+N)-mm-g,
                     us*IS/(kis+IS)-ms-msm*M,
                     up*IP/(kip+IP)*N/(knp+N)-mp-gp))


def scalar_jac(z, a):
    # J_log = D F(y) diag(y); no division by tiny biomasses.
    H,Nwc,I0,kw,kp,km,ks,um,us,up,kim,kis,kip,knm,knp,mm,ms,mp,gp,gm,kg,gamma,beta,msm=a
    M,S,P=math.exp(z[0]),math.exp(z[1]),math.exp(z[2])
    att=H*(kw+kp*P)
    IM=I0*math.exp(-att-km*M/2)
    IS=I0*math.exp(-att-km*M-ks*S)
    IP=I0*math.exp(-att/2)
    N=(Nwc-M)/H-P
    lm,ls,lp=IM/(kim+IM),IS/(kis+IS),IP/(kip+IP)
    dlm=kim*IM/(kim+IM)**2; dls=kis*IS/(kis+IS)**2; dlp=kip*IP/(kip+IP)**2
    am=um*N/(knm+N)*dlm; bm=um*lm*knm/(knm+N)**2
    ap=up*N/(knp+N)*dlp; bp=up*lp*knp/(knp+N)**2
    x=S/kg
    dg=gm*(1+gamma)*beta*x**(beta-1)/(2*kg*(1+gamma*x**beta)**2)
    return np.array(((-M*(am*km/2+bm/H),-S*dg,-P*(am*H*kp+bm)),
                     (-M*(us*dls*km+msm),-S*us*dls*ks,-P*us*dls*H*kp),
                     (-M*bp/H,0.,-P*(ap*H*kp/2+bp))))


# Optional acceleration. No fastmath, no changed precision or equations.
try:
    from numba import njit
    compiled_rhs=njit(cache=True)(scalar_rhs)
    compiled_jac=njit(cache=True)(scalar_jac)
except Exception:
    compiled_rhs=compiled_jac=None


def make_kernel(p, prefer_compiled=True):
    a=coefficients(p)
    f,j=scalar_rhs,scalar_jac
    backend='scalar_lsoda'
    if prefer_compiled and compiled_rhs is not None:
        try:
            z=np.log([max(p.N_wc/4,1e-10),1.,max(p.N_wc/(4*p.H),1e-10)])
            compiled_rhs(z,a);compiled_jac(z,a)
            f,j=compiled_rhs,compiled_jac
            backend='numba_lsoda'
        except Exception as exc:
            print('Numba unavailable; using scalar LSODA:',exc,flush=True)
    # Guard against changes to the authoritative shared equations. Check both
    # RHS and analytical log Jacobian at feasible and very small states.
    rng=np.random.default_rng(917)
    for q in rng.uniform(.001,.98,(16,3)):
        m=p.N_wc*q[0]; phy=(p.N_wc-m)*q[1]/p.H
        y=np.array([max(m,1e-20),max(q[2]*200,1e-20),max(phy,1e-20)])
        z=np.log(y);y=np.exp(z)
        F=per_capita(y,p)
        J=(jacobian(y,p)-np.diag(F))*y[None,:]/y[:,None]
        np.testing.assert_allclose(f(z,a),F,rtol=2e-11,atol=2e-12)
        np.testing.assert_allclose(j(z,a),J,rtol=2e-10,atol=2e-11)
    return f,j,a,backend


_CONTEXT=None


def init_basin(p,targets,days,mode):
    global _CONTEXT
    init_worker(p,targets,days)
    _CONTEXT=make_kernel(p) if mode!='reference' else None


def evaluate_basin(item):
    i,state=item
    if _CONTEXT is None:
        row=classify_seed((tuple(state),0,0))
        backend='reference_lsoda'
    else:
        f,j,a,backend=_CONTEXT
        def field(t,z):return f(z,a)
        def integrate(fun,span,z,**kwargs):
            return solve_ivp(fun,span,z,jac=lambda t,y:j(y,a),**kwargs)
        try:
            row=classify_seed((tuple(state),0,0),keep_path=False,field=field,solver=integrate)
        except (ValueError, ArithmeticError, RuntimeError, np.linalg.LinAlgError):
            row=classify_seed((tuple(state),0,0),keep_path=False)
            backend='reference_fallback'
        if row['label']==-1 and row['reason'] not in ('integration horizon reached','ambiguous attractor assignment'):
            # Retry numerical failures with the unchanged reference solver.
            reference=classify_seed((tuple(state),0,0),keep_path=False)
            if reference['label']>=0:
                row=reference;backend='reference_fallback'
    row.pop('path',None);row.pop('s_layer',None);row.pop('p_layer',None)
    row['backend']=backend
    return i,row


def evaluate_batch(items):
    return [evaluate_basin(item) for item in items]


class BasinPool:
    """Reuse workers and compiled kernels across all octree refinement stages."""
    def __init__(self,p,targets,days,workers,mode,checkpoint):
        from concurrent.futures import ProcessPoolExecutor
        self.start=__import__('time').monotonic();self.done=0;self.workers=workers
        self.checkpoint=checkpoint;self.pool=None
        if workers==1:init_basin(p,targets,days,mode)
        else:self.pool=ProcessPoolExecutor(max_workers=workers,initializer=init_basin,initargs=(p,targets,days,mode))

    def __enter__(self):return self
    def __exit__(self,*exc):
        if self.pool:self.pool.shutdown(wait=True,cancel_futures=True)

    def evaluate(self,states,stage):
        from concurrent.futures import wait,FIRST_COMPLETED
        import time
        n=len(states);out=[None]*n
        # Small batches reduce IPC, without assigning all hard trajectories to one worker.
        size=max(1,min(8,n//max(1,4*self.workers)))
        batches=[list(enumerate(states[i:i+size],i)) for i in range(0,n,size)]
        def progress():self.checkpoint(self.done,stage,time.monotonic()-self.start)
        if self.pool is None:
            for batch in batches:
                progress()
                for i,row in evaluate_batch(batch):out[i]=row;self.done+=1
        else:
            pending=set();next_batch=0
            try:
                while next_batch<len(batches) or pending:
                    progress()
                    while next_batch<len(batches) and len(pending)<2*self.workers:
                        pending.add(self.pool.submit(evaluate_batch,batches[next_batch]));next_batch+=1
                    ready,pending=wait(pending,timeout=.4,return_when=FIRST_COMPLETED)
                    for future in ready:
                        for i,row in future.result():out[i]=row;self.done+=1
            finally:
                for f in pending:f.cancel()
        progress();return out
