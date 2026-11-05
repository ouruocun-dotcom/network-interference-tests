"""Continuous WS rewiring scan with >=200 outcomes/graph and 80%-power efficiency.

Linear Hard2 response; fixed expected degree 8.
Rewiring changes several graph properties: scalar regressions are descriptive,
not causal partial effects of clustering. Same outcome/noise/draws over theta.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse,stats
import revision_study as r

SEED=2026100924
PROBABILITIES=(0.,.01,.02,.05,.1,.2,.5,1.)
THETA=(.15,.3,.5,.75,1.,1.5,2.,3.,4.,6.)

def reject(weights,draws):
    sd=np.sqrt(.24)*np.linalg.norm(weights,axis=1)
    z=np.matmul(draws-.4,weights)
    z=np.divide(z,sd[:,None,:],out=np.zeros_like(z),where=sd[:,None,:]>1e-12)
    return r.rank_p(np.max(np.abs(z),axis=2))<=.05

def threshold(power,target=.8):
    x=np.array(THETA);y=np.maximum.accumulate(power)
    if y[-1]<target:return np.nan
    if y[0]>=target:return np.nan  # left-censored, do not report a false threshold
    j=int(np.searchsorted(y,target))
    return float(x[j-1]+(target-y[j-1])*(x[j]-x[j-1])/(y[j]-y[j-1]))

def run(output,quick=False):
    start=time.perf_counter();rng=np.random.default_rng(SEED)
    n=80 if quick else 240;graphs=1 if quick else 6
    reps=4 if quick else 256;b=39 if quick else 499
    records=[];metric=[]
    for p in PROBABILITIES:
        for gi in range(graphs):
            gs=SEED+104729*gi
            g=nx.watts_strogatz_graph(n,8,p,seed=gs)
            assert nx.is_connected(g)
            a=nx.to_numpy_array(g,nodelist=range(n))
            split=np.random.default_rng(gs+19).permutation(n)
            f,c=np.sort(split[:n//2]),np.sort(split[n//2:])
            w,eig,u,k,groups,orth,tails=r.prepare(a,f,c)
            distance=sparse.csgraph.shortest_path(sparse.csr_matrix(a),directed=False)
            h=r.scale((distance<=2).astype(float)[np.ix_(f,c)])
            bank={method:k[groups[method]] for method in ('onehop','raw_max','orth_max','orth_degree_max')}
            bank['oracle']=h[None,:,:]
            draws=rng.binomial(1,.4,(reps,b+1,len(c))).astype(float)
            noise=rng.normal(size=(reps,len(f)))
            signal=(draws[:,0,:]-.4)@h.T
            noise_weights={name:np.einsum('rf,kfa->rak',noise,kk,optimize=True) for name,kk in bank.items()}
            signal_weights={name:np.einsum('rf,kfa->rak',signal,kk,optimize=True) for name,kk in bank.items()}
            for theta in THETA:
                rejections={name:reject(noise_weights[name]+theta*signal_weights[name],draws) for name in bank}
                for name,rr in rejections.items():
                    records.append({'p':p,'graph':gi,'seed':gs,'theta':theta,'method':name,
                                    'rate':rr.mean(),'gain':np.mean(rr.astype(int)-rejections['onehop']),
                                    'reps':reps,'B':b})
            metric.append({'p':p,'graph':gi,'clustering':nx.average_clustering(g),
                           'degree_cv':a.sum(axis=1).std()/8,
                           'gap':1-eig[-2],'coverage_hard2':np.mean(distance<=2),
                           'projection_share':np.sum(tails[0]*w[np.ix_(f,c)])**2/
                                              (np.sum(tails[0]**2)*np.sum(w[np.ix_(f,c)]**2))})
            print('controlled',p,gi+1,'/',graphs,flush=True)
    output.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(records)
    frame.to_csv(output/'round2_controlled_rates.csv',index=False)
    pd.DataFrame(metric).to_csv(output/'round2_controlled_geometry.csv',index=False)
    rows=[];eff=[]
    for key,part in frame.groupby(['p','theta','method']):
        mean,lo,hi=r.cluster_ci(part.rate);gain,glo,ghi=r.cluster_ci(part.gain)
        rows.append(dict(zip(['p','theta','method'],key),rate=mean,ci_low=max(0,lo),ci_high=min(1,hi),
                         gain=gain,gain_ci_low=glo,gain_ci_high=ghi))
    for (p,gi),part in frame.groupby(['p','graph']):
        curves=part.pivot(index='theta',columns='method',values='rate').reindex(THETA)
        t={name:threshold(curves[name].to_numpy()) for name in curves}
        for name,value in t.items():
            eff.append({'p':p,'graph':gi,'method':name,'theta80':value,
                        'relative_efficiency':t['onehop']/value,
                        'oracle_relative_strength':value/t['oracle']})
    pd.DataFrame(rows).to_csv(output/'round2_controlled_summary.csv',index=False)
    efficiency=pd.DataFrame(eff)
    efficiency.to_csv(output/'round2_controlled_efficiency.csv',index=False)
    intervals=[]
    for (pp,name),part in efficiency.groupby(['p','method']):
        for quantity in ('theta80','relative_efficiency','oracle_relative_strength'):
            observed=part[quantity].dropna().to_numpy()
            if len(observed):
                mean,lo,hi=r.cluster_ci(observed)
                intervals.append({'p':pp,'method':name,'quantity':quantity,'mean':mean,
                                  'ci_low':lo,'ci_high':hi,'uncensored_graphs':len(observed),
                                  'total_graphs':graphs})
    pd.DataFrame(intervals).to_csv(output/'round2_controlled_efficiency_intervals.csv',index=False)
    (output/'round2_controlled_run.json').write_text(json.dumps({'seed':SEED,'graphs':graphs,'reps':reps,
      'B':b,'n':n,'p':PROBABILITIES,'theta':THETA,'seconds':time.perf_counter()-start,
      'threshold':'monotone-envelope linear interpolation, censored values remain missing'},indent=2))
    print('completed',time.perf_counter()-start,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'output_round2')
    args=p.parse_args();run(args.output,args.quick)
