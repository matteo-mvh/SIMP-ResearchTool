"""Synthetic two-dimensional state map, not real continuation."""
import math
import sys
import time
from app.core.control import load_args,StopRequested
from app.core.demo_helpers import linspace,csv_write,plt_module,finish_plot

def main():
    p,ctx,_=load_args()
    n=p['resolution']
    x=linspace(p['nwc_min'],p['nwc_max'],n)
    y=linspace(p['kw_min'],p['kw_max'],n)
    rows=[]; z=[]
    try:
        for j,kw in enumerate(y):
            line=[]
            for i,nwc in enumerate(x):
                v=1 if (0.25 < (nwc-p['nwc_min'])/(max(1e-9,p['nwc_max']-p['nwc_min'])) < 0.7 and
                         kw < 0.2+0.65*(nwc-p['nwc_min'])/max(1e-9,p['nwc_max']-p['nwc_min'])) else 0
                line.append(v)
                rows.append({'Nwc':round(nwc,5),'kw':round(kw,5),'bistable_demo':v})
                ctx.checkpoint((j*n+i+1)/(n*n)*88,f'Grid {j*n+i+1}/{n*n}')
                time.sleep(p['delay_ms']/1000)
            z.append(line)
        csv_write(ctx.folder/'state_map.csv',list(rows[0]),rows)
        ctx.checkpoint(93,'Plotting map',True)
        plt=plt_module();fig,ax=plt.subplots(figsize=(7.5,4))
        im=ax.imshow(z,origin='lower',aspect='auto',interpolation='nearest',
                     extent=[min(x),max(x),min(y),max(y)],cmap='Blues',vmin=0,vmax=1)
        fig.colorbar(im,ax=ax,ticks=[0,1],label='Synthetic two-state indicator')
        ax.set(xlabel='Total nitrogen inventory Nwc',ylabel='Background attenuation kw',
               title='2D state map — placeholder')
        fig.tight_layout();finish_plot(fig,ctx,'continuation_2d.png')
        ctx.progress(100,'Complete',True)
    except StopRequested: return 130
    return 0
if __name__=='__main__':sys.exit(main())
