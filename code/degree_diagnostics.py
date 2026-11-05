"""Outcome-free WS degree scan using the manuscript's archived graph seeds."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import networkx as nx
from network_diagnostics import response_operator,predict_threshold

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent if HERE.name=='code' else HERE
rows=[]
for degree in [4,8,16]:
    for probability in [0,.01,.02,.05,.1,.2,.5,1]:
        for graph in range(6):
            seed=2026100933+104729*graph
            for attempt in range(1000):
                g=nx.watts_strogatz_graph(240,degree,probability,seed=seed+981*attempt)
                if nx.is_connected(g):break
            else:raise RuntimeError('Connectivity rejection exhausted')
            a=nx.to_numpy_array(g,nodelist=range(240))
            order=np.random.default_rng(seed+19).permutation(240)
            f,c=np.sort(order[:120]),np.sort(order[120:])
            h=response_operator(a,f,c,'twohop')
            for method,kernel in [('onehop','onehop'),('matched_kernel','matched')]:
                row=predict_threshold(a,f,c,h,kernel=kernel)
                rows.append(dict(source_table='ws_degree_diagnostics',degree=degree,p=probability,
                    graph=graph,seed=seed,attempts=attempt+1,method=method,
                    clustering=nx.average_clustering(g),graph_sha256=hashlib.sha256(a.astype('uint8').tobytes()).hexdigest(),**row))
out=pd.DataFrame(rows)
out.to_csv(ROOT/'degree_diagnostics.csv',index=False)
print(out.query('method=="onehop"').groupby(['degree','p']).agg(gamma=('gamma_infinity','mean'),
    cosine=('cosine','mean'),finite=('theta','count')).to_string())
