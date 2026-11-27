"""Placeholder branches for GUI testing; not a SIMP equilibrium computation."""
import math
import sys
import time
from app.core.control import load_args,StopRequested
from app.core.demo_helpers import linspace,csv_write,plt_module,finish_plot

def main():
    p,ctx,_=load_args()
    x=linspace(p['start'],p['end'],p['points']); rows=[]
    try:
        for i,z in enumerate(x):
            u=i/max(1,len(x)-1)
            high=max(0.,155-110*u+12*math.sin(math.pi*u))
            low=12+17*u
            rows.append({'parameter':round(z,6),'S_high':round(high,5),'S_low':round(low,5)})
            ctx.checkpoint((i+1)/len(x)*87,f'Computing point {i+1}/{len(x)}')
            time.sleep(p['delay_ms']/1000)
        csv_write(ctx.folder/'branches.csv',list(rows[0]),rows)
        ctx.checkpoint(93,'Plotting branches',True)
        plt=plt_module(); fig,ax=plt.subplots(figsize=(7.5,4))
        ax.plot(x,[r['S_high'] for r in rows],label='High branch (demo)',linewidth=2.2)
        ax.plot(x,[r['S_low'] for r in rows],label='Low branch (demo)',linewidth=2.2)
        ax.set(xlabel=p['parameter'],ylabel='Illustrative seagrass biomass',title='1D continuation — placeholder')
        ax.legend(); ax.grid(alpha=.2);fig.tight_layout()
        finish_plot(fig,ctx,'continuation_1d.png');ctx.progress(100,'Complete',True)
    except StopRequested: return 130
    return 0
if __name__=='__main__': sys.exit(main())
