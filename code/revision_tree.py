"""Local-tree alignment ratios on random regular graphs."""
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import revision_study as r

def run():
    rows=[]
    for n in (240,1000,5000):
        for d in (4,6,8,12):
            for gi in range(10):
                seed=2026100926+n+104729*gi+1009*d
                g=nx.random_regular_graph(d,n,seed=seed)
                a=nx.to_scipy_sparse_array(g,nodelist=range(n),dtype=float,format='csr')
                split=np.random.default_rng(seed+19).permutation(n)
                f,c=np.sort(split[:n//2]),np.sort(split[n//2:])
                w=a/d;l=w[f,:][:,c].tocsr();rr=(w@w)[f,:][:,c].tocsr()
                h=(a+a@a)[f,:][:,c].tocsr();h.data[:]=1.
                h=h*np.sqrt(len(f)/h.nnz)
                base=float(l.multiply(h).sum())/np.sqrt(float(l.multiply(l).sum()))
                alpha=float(rr.multiply(l).sum())/float(l.multiply(l).sum())
                for name,k in [('two_step',rr),('projected_two_step',rr-alpha*l)]:
                    gamma=float(k.multiply(h).sum())/np.sqrt(float(k.multiply(k).sum()))
                    rows.append({'n':n,'degree':d,'graph':gi,'method':name,
                                 'ratio':gamma/base,'tree_limit':np.sqrt(d-1),
                                 'clustering':nx.average_clustering(g)})
    output=Path(__file__).parent/'output_round2';output.mkdir(exist_ok=True)
    frame=pd.DataFrame(rows);frame.to_csv(output/'round2_tree_graph_ratios.csv',index=False)
    summary=[]
    for key,part in frame.groupby(['n','degree','method']):
        mean,lo,hi=r.cluster_ci(part.ratio)
        summary.append(dict(zip(['n','degree','method'],key),ratio=mean,ci_low=lo,ci_high=hi,
                            tree_limit=part.tree_limit.iloc[0]))
    pd.DataFrame(summary).to_csv(output/'round2_tree_summary.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False))

if __name__=='__main__':run()
