"""Integrate the established SIMP RHS; export only areal biomass/resources."""
from dataclasses import replace
import json
import time
import numpy as np
from scipy.integrate import solve_ivp
from model.shared_eqn import rhs, jacobian
from model.timeseries_config import parse_job
from app.core.control import load_args, StopRequested


def integrate(p, initial, duration, samples, checkpoint=lambda fraction:None):
    # M,S are already areal. Only phytoplankton is volumetric internally.
    state=np.array([initial[0],initial[1],initial[2]/p.H])
    times=np.linspace(0,duration,samples)
    chunks=[]
    last=0
    # Bounded integration intervals allow cooperative pause/stop and progress.
    for end in np.unique(np.linspace(1,samples-1,min(100,samples-1),dtype=int)):
        checkpoint(last/(samples-1))
        sol=solve_ivp(rhs,(times[last],times[end]),state,args=(p,),method='BDF',
                      jac=lambda t,y,p:jacobian(y,p),t_eval=times[last:end+1],rtol=1e-7,atol=1e-9)
        if not sol.success: raise RuntimeError(sol.message)
        chunks.append(sol.y if last==0 else sol.y[:,1:])
        state=sol.y[:,-1];last=end
    y=np.concatenate(chunks,axis=1)
    area=np.vstack((y[0],y[1],y[2]*p.H,p.N_wc-y[0]-y[2]*p.H))
    if not np.all(np.isfinite(area)) or np.min(area)<-1e-5:
        raise RuntimeError('Integration left the physical nonnegative domain; inspect initial values and settings')
    return times,area


def run(values,ctx):
    ctx.checkpoint(0,'Preparing time integration',force=True)
    from scripts.continuation_plots import plt,save,VARIABLE_COLORS
    p,depths,groups,initial,v=parse_job(values)
    started=time.monotonic();datasets=[]
    for i,h in enumerate(depths):
        t,y=integrate(replace(p,H=h),initial,float(v['duration']),int(v['samples']),
                      lambda f:ctx.checkpoint(85*(i+f)/len(depths),f'Integrating depth {h:g} m'))
        datasets.append((h,t,y))
        np.savetxt(ctx.folder/f'timeseries_H{h:g}.csv',np.column_stack((t,y.T)),delimiter=',',
                   header='time_days,M_area,S_area,P_area,N_area',comments='')
    for index,group in enumerate(groups):
        ctx.checkpoint(85+14*index/len(groups),f'Plotting {group}',force=True)
        rows=2 if len(depths)==4 else 1;cols=2 if len(depths)>1 else 1
        fig,axes=plt.subplots(rows,cols,figsize=(6*cols,4*rows),squeeze=False,sharex=True,sharey=True)
        for ax,(h,t,y) in zip(axes.flat,datasets):
            for key in group:
                ax.plot(t,y['MSPN'.index(key)],label=key,color=VARIABLE_COLORS[key if key in 'MS' else key+'_area'])
            ax.set(title=f'H = {h:g} m',xlabel='Time (days)',ylabel='Areal amount (mmol N/m²)')
            ax.grid(alpha=.2);ax.legend(frameon=False)
        fig.tight_layout();save(fig,ctx.folder/'figures',f'timeseries_{index+1:02d}_{group}')
    (ctx.folder/'summary.json').write_text(json.dumps({'seconds':time.monotonic()-started,
        'depths':depths,'plot_groups':groups,'equation_source':'model/shared_eqn.py','units':'mmol N/m²'},indent=2))
    ctx.checkpoint(100,'Completed time series',force=True)


def main():
    values,ctx,_=load_args()
    try:run(values,ctx)
    except StopRequested:return 130
    return 0
