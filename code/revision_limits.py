"""Two-component local-power calculation and Gaussian-limit comparison."""
from pathlib import Path
import argparse
import time
import json
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse, stats
import revision_study as r

SEED=2026100941
T=(0.,2.,4.,6.,8.)

def formula(t,d):
    c=stats.norm.ppf((1+np.sqrt(.95))/2)
    means=t*np.sqrt(.24/d)*np.array([1,np.sqrt(d-1)])
    return 1-np.prod(stats.norm.cdf(c-means)-stats.norm.cdf(-c-means))

def run(out,quick=False):
    started=time.perf_counter();rng=np.random.default_rng(SEED)
    rows=[];geometry=[]
    graphs=1 if quick else 6; reps=8 if quick else 512
    B=39 if quick else 499
    for n in ((100,200) if quick else (1000,5000)):
        for d in (4,8):
            for gi in range(graphs):
                seed=int(rng.integers(0,2**31-1))
                g=nx.random_regular_graph(d,n,seed=seed)
                perm=np.random.default_rng(seed+104729).permutation(n)
                f,c=np.sort(perm[:n//2]),np.sort(perm[n//2:])
                a=nx.to_scipy_sparse_array(g,nodelist=range(n),dtype=float,format='csr')
                w=a/d; L=w[f,:][:,c].tocsr();R=(w@w)[f,:][:,c].tocsr()
                support=(a+a@a).tocsr();support.data[:]=1
                H=support[f,:][:,c].tocsr();H=H*np.sqrt(len(f)/H.nnz)
                ll=float(L.multiply(L).sum());rr=float(R.multiply(R).sum())
                lg=float(L.sum())/n;rg=float(R.sum())/n
                gg=len(f)*len(c)/n**2
                alpha,beta=np.linalg.solve([[ll,lg],[lg,gg]],
                                           [float(L.multiply(R).sum()),rg])
                removed=alpha*float(L.multiply(R).sum())+beta*rg
                geometry.append(dict(n=n,d=d,graph=gi,seed=seed,
                    constant_share=rg*rg/(gg*rr),controlled_removed_share=removed/rr,
                    energy_R=rr,energy_L=ll,graphs=graphs))
                count=np.zeros((len(T),2),dtype=int)
                for start in range(0,reps,16):
                    batch=min(16,reps-start)
                    draws=rng.binomial(1,.4,(batch,B+1,len(c))).astype(float)
                    U=draws-.4
                    noise=rng.normal(size=(batch,len(f)))
                    signal=np.asarray(H@U[:,0,:].T).T
                    def weights(y):
                        l=np.asarray(L.T@y.T).T;rrr=np.asarray(R.T@y.T).T
                        glob=y.sum(axis=1)[:,None]/n
                        return np.array([l,rrr,rrr-alpha*l-beta*glob])
                    wn,ws=weights(noise),weights(signal)
                    dn=np.einsum('rba,kra->rbk',U,wn,optimize=True)
                    ds=np.einsum('rba,kra->rbk',U,ws,optimize=True)
                    nn=(wn*wn).sum(axis=2);ns=(wn*ws).sum(axis=2);ss=(ws*ws).sum(axis=2)
                    for ti,t in enumerate(T):
                        theta=t/np.sqrt(len(f))
                        sd=np.sqrt(.24*np.maximum(0,nn+2*theta*ns+theta**2*ss)).T
                        z=np.divide(dn+theta*ds,sd[:,None,:],out=np.zeros_like(dn),
                                    where=sd[:,None,:]>1e-12)
                        for mi,idx in enumerate(([0,1],[0,2])):
                            count[ti,mi]+=np.sum(r.rank_p(np.max(np.abs(z[:,:,idx]),axis=2))<=.05)
                for ti,t in enumerate(T):
                    for mi,method in enumerate(('raw_two','controlled_two')):
                        rows.append(dict(n=n,d=d,graph=gi,seed=seed,t=t,
                            theta=t/np.sqrt(len(f)),method=method,rate=count[ti,mi]/reps,
                            rejects=count[ti,mi],replicates=reps,B=B,theory=formula(t,d)))
                out.mkdir(parents=True,exist_ok=True)
                pd.DataFrame(rows).to_csv(out/'round4_local_graph_rates.csv',index=False)
                print('local',n,d,gi+1,'/',graphs,flush=True)
    frame=pd.DataFrame(rows);summary=[]
    for key,q in frame.groupby(['n','d','t','method']):
        rate,lo,hi=r.cluster_ci(q.rate)
        summary.append(dict(zip(['n','d','t','method'],key),rate=rate,
            ci_low=max(0,lo),ci_high=min(1,hi),theory=q.theory.iloc[0],
            discrepancy=rate-q.theory.iloc[0],graphs=graphs,replicates=reps,B=B))
    pd.DataFrame(summary).to_csv(out/'round4_local_summary.csv',index=False)
    pd.DataFrame(geometry).to_csv(out/'round4_local_geometry.csv',index=False)
    (out/'round4_local_run.json').write_text(json.dumps(dict(seed=SEED,
        seconds=time.perf_counter()-started,graphs=graphs,replicates=reps,B=B),indent=2))
    print('completed',time.perf_counter()-started,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'output_round4')
    args=p.parse_args();run(args.output,args.quick)
