"""Synthetic time series, not the SIMP model."""
import math
import time
import sys
from app.core.control import load_args,StopRequested
from app.core.demo_helpers import csv_write,plt_module,finish_plot

def main():
    p,ctx,_=load_args();rows=[]
    try:
        for day in range(p['days']):
            value=80+20*math.exp(-day/45)*math.cos(day/12)-35*math.tanh((day-80)/50)*p['kw']
            rows.append({'day':day,'biomass_demo':round(max(0,value),6)})
            ctx.checkpoint((day+1)/p['days']*90,f'Day {day+1}/{p["days"]}')
            time.sleep(p['delay_ms']/1000)
        csv_write(ctx.folder/'timeseries.csv',list(rows[0]),rows)
        ctx.checkpoint(94,'Generating figure',True)
        plt=plt_module();fig,ax=plt.subplots(figsize=(7.5,4))
        ax.plot([r['day'] for r in rows],[r['biomass_demo'] for r in rows])
        ax.set(xlabel='Day',ylabel='Synthetic biomass',title='Time-series — placeholder')
        ax.grid(alpha=.25);fig.tight_layout();finish_plot(fig,ctx,'timeseries.png')
        ctx.progress(100,'Complete',True)
    except StopRequested: return 130
    return 0
if __name__=='__main__':sys.exit(main())
