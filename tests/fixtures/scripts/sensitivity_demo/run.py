"""Demonstration sweep; streaming writes make 10,000+ seeds practical.

NOT a SIMP solver. An actual sensitivity script can retain this output
contract while replacing the synthetic score with scientific calculations.
"""
from __future__ import annotations
import csv
import math
import random
import sys
import time
from app.core.control import load_args, StopRequested
from app.core.demo_helpers import plt_module, finish_plot


def main():
    p,ctx,job=load_args()
    n=p['sample_count']; rng=random.Random(p['random_seed'])
    plot_sample=[]                         # Bounded memory for preview figures
    summary_counts=[0,0]
    update_every=max(1,min(200,n//100))    # No per-seed GUI overhead for large sweeps
    try:
        with (ctx.folder/'samples.csv').open('w',newline='',encoding='utf-8') as sfile, \
             (ctx.folder/'seeds.csv').open('w',newline='',encoding='utf-8') as seedfile:
            samples=csv.DictWriter(sfile,fieldnames=[
                'seed_id','kw','g_M','m_SM','mu_M','bistability','delta_S'])
            seeds=csv.DictWriter(seedfile,fieldnames=[
                'seed_id','kw','g_M','m_SM','mu_M','metric__bistability','metric__delta_S'])
            samples.writeheader();seeds.writeheader()
            for i in range(n):
                kw=max(0.0,p['kw']*(0.65+0.7*rng.random()))
                g_m=0.01+0.09*rng.random()
                m_sm=0.00001+0.00008*rng.random()
                mu_m=0.17+0.38*rng.random()
                score=0.50+2.2*g_m-3000*m_sm-0.38*kw+0.08*rng.gauss(0,1)
                bistable=int(score>0.45)
                value=round(30+120/(1+math.exp(-7*(score-0.45))),2)
                seed_id=f"S-{job['id'][:8]}-{i+1:06d}"
                record={'seed_id':seed_id,'kw':round(kw,5),'g_M':round(g_m,5),
                        'm_SM':round(m_sm,8),'mu_M':round(mu_m,5)}
                samples.writerow({**record,'bistability':bistable,'delta_S':value})
                seeds.writerow({**record,'metric__bistability':bistable,'metric__delta_S':value})
                summary_counts[bistable]+=1
                # Reservoir sampling retains representative points without holding all seeds in RAM.
                point=(kw,value,bistable)
                if len(plot_sample)<3000:plot_sample.append(point)
                else:
                    index=rng.randrange(i+1)
                    if index<len(plot_sample):plot_sample[index]=point
                if (i+1)%update_every==0 or i+1==n:
                    ctx.checkpoint((i+1)/n*85,f'Samples {i+1:,}/{n:,}')
                if p['delay_ms']:
                    time.sleep(p['delay_ms']/1000)
        ctx.checkpoint(90,'Creating demo figures',True)
        plt=plt_module()
        fig,ax=plt.subplots(figsize=(7.5,4))
        ax.scatter([x[0] for x in plot_sample],[x[1] for x in plot_sample],
                   c=[x[2] for x in plot_sample],s=12,alpha=.7,cmap='coolwarm')
        ax.set(xlabel='Background attenuation kw (demo)',ylabel='Synthetic seagrass contrast',
               title=f'Sensitivity — placeholder data ({n:,} seeds)')
        fig.tight_layout();finish_plot(fig,ctx,'sensitivity_scatter.png')
        fig,ax=plt.subplots(figsize=(7.5,4))
        ax.hist([x[1] for x in plot_sample],bins=25,color='#16803F',edgecolor='white')
        ax.set(xlabel='Synthetic seagrass contrast',ylabel='Preview frequency',
               title='Preview sample (max 3,000 points)')
        fig.tight_layout();finish_plot(fig,ctx,'sensitivity_histogram.png')
        ctx.progress(100,f'Complete — {n:,} seeds saved',True)
    except StopRequested:
        print('Stopped at checkpoint; partial CSVs were retained',flush=True)
        return 130
    return 0


if __name__=='__main__':sys.exit(main())
