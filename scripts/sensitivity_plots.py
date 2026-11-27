"""Full-sample marginal rates and bounded representative diagnostic plots."""
import csv
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def plot_results(x,y,keys,ranges,reservoir,summary,ctx,seed):
    # Deliberately independent of continuation modules and pyoomph.
    folder=ctx.folder/'figures';folder.mkdir(exist_ok=True)
    figure_count=0
    def save(fig,name):
        nonlocal figure_count
        fig.tight_layout();fig.savefig(folder/f'{name}.png',dpi=220,bbox_inches='tight')
        fig.savefig(folder/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
        figure_count+=1
        ctx.checkpoint(min(99,85+14*figure_count/(8+2*len(keys))),f'Plotting {name}',force=True)
    idx=np.sort(np.asarray(reservoir,dtype=int));sx=np.asarray(x[idx]);sy=np.asarray(y[idx])
    valid=np.isfinite(sy[:,0]);b=sy[:,0]==1
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    counts=[summary['eligible_samples']-summary['bistable_samples'],summary['bistable_samples'],summary['marginal_samples'],summary['solver_errors']]
    axes[0].bar(['<2 stable','≥2 stable','Marginal','Error'],counts,color=['#608aab','#c84d43','#d9aa40','#666666'])
    probability=summary['bistability_probability']
    caption=f'{probability:.1%} detected bistability' if probability is not None else 'No classifiable samples'
    axes[0].set(ylabel='Parameter sets',title=caption+'\n(all eligible samples; marginal/errors shown separately)')
    eligible=np.isfinite(y[:,0]);cum=np.cumsum(eligible);wins=np.cumsum(np.nan_to_num(y[:,0]))
    take=np.unique(np.linspace(0,len(y)-1,min(2000,len(y)),dtype=int))
    axes[1].plot(take+1,np.divide(wins[take],cum[take],out=np.full(len(take),np.nan),where=cum[take]>0))
    axes[1].set(xlabel='LHS sample index',ylabel='Detected bistability fraction',ylim=(0,1),title='Convergence diagnostic (not a confidence interval)')
    save(fig,'01_outcomes_and_convergence')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    deltas=sy[b,1];deltas=deltas[np.isfinite(deltas)]
    if len(deltas):
        axes[0].hist(deltas,bins=30,color='#278360');ordered=np.sort(deltas)
        axes[1].plot(ordered,np.arange(1,len(ordered)+1)/len(ordered))
    else:
        for ax in axes[:2]:ax.text(.5,.5,'No detected bistability',ha='center',transform=ax.transAxes)
    axes[0].set(xlabel='ΔS (mmol N/m²)',ylabel='Sample count');axes[1].set(xlabel='ΔS (mmol N/m²)',ylabel='Cumulative fraction')
    axes[2].hist(sy[valid,2],bins=np.arange(0,max(3,np.max(sy[:,2])+2))-.5)
    axes[2].set(xlabel='Number of strictly stable states',ylabel='Sample count')
    fig.suptitle(f'Representative diagnostics (up to {len(idx):,} samples)')
    save(fig,'02_delta_S_and_stable_counts')
    if not keys:return
    bins=12;rates=[];mean_delta=[];support=[]
    with (ctx.folder/'conditional_bins.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['parameter','lower','upper','eligible','bistable','probability','mean_delta_S_bistable'])
        for j,key in enumerate(keys):
            lo,hi=ranges[key];edges=np.linspace(lo,hi,bins+1)
            column=np.asarray(x[:,j]);indices=np.minimum(bins-1,np.maximum(0,((column-lo)/(hi-lo)*bins).astype(int)))
            ok=np.isfinite(y[:,0]);bi=y[:,0]==1
            counts=np.bincount(indices[ok],minlength=bins);hits=np.bincount(indices[bi],minlength=bins)
            sums=np.bincount(indices[bi],weights=y[bi,1],minlength=bins)
            rate=np.divide(hits,counts,out=np.full(bins,np.nan),where=counts>0)
            delta=np.divide(sums,hits,out=np.full(bins,np.nan),where=hits>0)
            rates.append(rate);mean_delta.append(delta);support.append(counts)
            for k in range(bins):writer.writerow([key,edges[k],edges[k+1],counts[k],hits[k],rate[k],delta[k]])
            fig,axes=plt.subplots(2,2,figsize=(10,7))
            centers=(edges[1:]+edges[:-1])/2
            axes[0,0].plot(centers,rate,'o-',color='#b72535');axes[0,0].set(ylabel='P(detected bistability | bin)',ylim=(0,1))
            axes[0,1].bar(centers,counts,width=(hi-lo)/bins*.9);axes[0,1].set(ylabel='Eligible samples per bin')
            axes[1,0].scatter(sx[b,j],sy[b,1],s=5,alpha=.25,color='#278360',rasterized=True)
            axes[1,0].plot(centers,delta,color='black');axes[1,0].set(ylabel='ΔS (mmol N/m²); black = full-bin mean')
            for value,color,label in ((0,'#4080aa','<2 stable'),(1,'#c63e38','≥2 stable')):
                subset=sx[sy[:,0]==value,j]
                if len(subset):axes[1,1].hist(subset,bins=edges,density=True,histtype='step',label=label,color=color)
            axes[1,1].legend();axes[1,1].set(ylabel='Conditional input density (representative)')
            for ax in axes.flat:ax.set_xlabel(key);ax.grid(alpha=.15)
            fig.suptitle(key+' — marginal effects; other sampled parameters averaged out')
            save(fig,f'parameter_{key}')
    for name,matrix,cmap,title in [('bistability_strips',rates,'Reds','Bistability probability (all eligible samples)'),
                                  ('delta_S_strips',mean_delta,'YlOrRd','Mean ΔS among bistable samples (mmol N/m²)'),
                                  ('bin_support',support,'Blues','Eligible sample counts')]:
        fig,ax=plt.subplots(figsize=(11,max(3,.38*len(keys))))
        image=ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',cmap=cmap,
                        vmin=0,vmax=1 if name=='bistability_strips' else None,extent=(0,1,len(keys)-.5,-.5))
        ax.set_yticks(range(len(keys)),[f'{k}  [{ranges[k][0]:g} … {ranges[k][1]:g}]' for k in keys])
        ax.set(xlabel='Position within each parameter range: minimum → maximum',title=title)
        fig.colorbar(image,ax=ax);save(fig,name)
    # Predictive sensitivity: held-out permutation importance, not causal effects/Sobol indices.
    from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import brier_score_loss,r2_score
    ranking=np.nanvar(np.asarray(rates),axis=1)
    scores={}
    for target,mask,values,kind in [('bistability',valid,sy[:,0],'classifier'),('delta_S',b,sy[:,1],'regressor')]:
        a=sx[mask];z=values[mask];fig,ax=plt.subplots(figsize=(9,max(3,.32*len(keys))))
        unique,counts=np.unique(z,return_counts=True)
        usable=len(z)>=40 and len(unique)>1 and (kind!='classifier' or counts.min()>=5)
        if usable:
            train,test=train_test_split(np.arange(len(z)),test_size=.3,random_state=seed,
                                        stratify=z if kind=='classifier' else None)
            klass=RandomForestClassifier if kind=='classifier' else RandomForestRegressor
            model=klass(n_estimators=120,max_depth=12,min_samples_leaf=3,n_jobs=1,random_state=seed)
            model.fit(a[train],z[train])
            importance=permutation_importance(model,a[test],z[test],n_repeats=5,random_state=seed,
                          scoring='neg_brier_score' if kind=='classifier' else 'neg_mean_squared_error',n_jobs=1)
            order=np.argsort(importance.importances_mean)
            ax.barh(np.array(keys)[order],importance.importances_mean[order],xerr=importance.importances_std[order])
            ax.set_xlabel('Held-out Brier error increase' if kind=='classifier' else 'Held-out squared ΔS error increase')
            score=(brier_score_loss(z[test],model.predict_proba(a[test])[:,1]) if kind=='classifier' else r2_score(z[test],model.predict(a[test])))
            baseline=(brier_score_loss(z[test],np.full(len(test),np.mean(z[train]))) if kind=='classifier' else 0.)
            reliable=score<baseline if kind=='classifier' else score>0
            scores[target]={'validation_metric':'Brier loss (lower better)' if kind=='classifier' else 'R²', 'value':float(score),
                            'baseline':float(baseline),'beats_constant_baseline':bool(reliable),
                            'samples':len(z),'importance':dict(zip(keys,map(float,importance.importances_mean)))}
            if kind=='classifier':ranking=importance.importances_mean
            ax.set_title(f'{target}: predictive sensitivity; validation score = {score:.3g}' +
                         ('' if reliable else '\nModel does not beat constant baseline — ranking unreliable'))
        else:
            ax.text(.5,.5,'Insufficient samples or outcome variation\nfor a defensible importance ranking',ha='center',va='center',transform=ax.transAxes)
            ax.set_title(target+' sensitivity')
        save(fig,'importance_'+target)
    summary['predictive_sensitivity']=scores
    if len(keys)>1:
        corr=np.array([[spearmanr(sx[:,i],sx[:,j]).statistic for j in range(len(keys))] for i in range(len(keys))])
        fig,ax=plt.subplots(figsize=(max(6,len(keys)*.35),max(5,len(keys)*.35)))
        im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(len(keys)),keys,rotation=90);ax.set_yticks(range(len(keys)),keys)
        ax.set_title('LHS input rank correlations (representative sample)');fig.colorbar(im,ax=ax);save(fig,'input_correlations')
        top=np.argsort(np.nan_to_num(ranking))[-min(6,len(keys)):]
        for i,j in combinations(top,2):
            fig,axes=plt.subplots(1,2,figsize=(11,4))
            for ax,mask,values,title,cmap in [(axes[0],valid,sy[:,0],'Bistability probability','Reds'),
                                            (axes[1],b,sy[:,1],'Mean ΔS among bistable samples','YlOrRd')]:
                if np.any(mask):
                    image=ax.hexbin(sx[mask,i],sx[mask,j],C=values[mask],reduce_C_function=np.mean,
                                    gridsize=12,mincnt=1,cmap=cmap)
                    fig.colorbar(image,ax=ax)
                else:ax.text(.5,.5,'No eligible samples',ha='center',transform=ax.transAxes)
                ax.set(xlabel=keys[i],ylabel=keys[j],title=title+'\n(representative sample; sparse cells uncertain)')
            save(fig,f'pair_{keys[i]}_{keys[j]}')
    summary['figure_count']=figure_count
