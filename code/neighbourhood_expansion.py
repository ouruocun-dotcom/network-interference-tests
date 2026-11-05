"""Distinct first- and second-neighbour counts, edge-weight dispersion and overlap."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.csgraph import shortest_path
from village_study import load_graphs
from fetch_village_data import fetch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent if HERE.name=='code' else HERE

def overlap_geometry(a,f,c):
    a=np.asarray(a,dtype=float);f=np.asarray(f,dtype=int);c=np.asarray(c,dtype=int)
    distance=shortest_path(a,directed=False)
    block=distance[np.ix_(f,c)]
    first=block==1;second=block==2
    m1=int(first.sum());m2=int(second.sum())
    if not m1:raise ValueError('No first-neighbour cross-pairs')
    degree=a.sum(axis=1)
    inv=np.divide(1,np.sqrt(degree),out=np.zeros_like(degree),where=degree>0)
    w=inv[:,None]*a*inv[None,:]
    local=w[np.ix_(f,c)];h=(first|second).astype(float)
    weights=local[first]
    cv2=float(np.var(weights)/np.mean(weights)**2)
    cosine=float(np.sum(local*h)/(np.linalg.norm(local)*np.linalg.norm(h)))
    ratio=m2/m1
    predicted=1/np.sqrt((1+ratio)*(1+cv2))
    z1=float(np.sum(distance==1)/len(a));z2=float(np.sum(distance==2)/len(a))
    return dict(first_pairs=m1,second_pairs=m2,block_z1=m1/len(f),block_z2=m2/len(f),
        expansion_ratio=ratio,global_z1=z1,global_z2=z2,global_expansion_ratio=z2/z1,
        excess_degree=float(np.sum(degree*(degree-1))/degree.sum()),
        edge_weight_cv_squared=cv2,cosine=cosine,
        uncorrected_prediction=1/np.sqrt(1+ratio),exact_prediction=predicted,
        corrected_overlap=cosine*np.sqrt(1+cv2),
        global_approximation=1/np.sqrt(1+z2/z1),
        identity_error=abs(predicted-cosine),clustering=nx.average_clustering(nx.from_numpy_array(a)))

def ensemble_graph(kind,n,seed):
    for attempt in range(1000):
        s=seed+981*attempt
        if kind=='ER':g=nx.fast_gnp_random_graph(n,8/(n-1),seed=s)
        elif kind=='Regular':g=nx.random_regular_graph(8,n,seed=s)
        elif kind=='BA':g=nx.barabasi_albert_graph(n,4,seed=s)
        elif kind.startswith('SBM'):
            mixing=.03 if kind.endswith('003') else .45;q=n//2
            g=nx.stochastic_block_model([q,n-q],[[8*(1-mixing)/(q-1),8*mixing/q],[8*mixing/q,8*(1-mixing)/(q-1)]],seed=s)
        else:g=nx.watts_strogatz_graph(n,8,{'WS_002':.02,'WS_020':.2,'WS_100':1.}[kind],seed=s)
        if nx.is_connected(g):return g
    raise RuntimeError('Connectivity rejection exhausted')

def run():
    df=pd.read_csv(ROOT/'aggregate_results.csv',low_memory=False)
    geometry=df[(df.source_table=='round3_confirmation_geometry')&(df.design=='Bernoulli')]
    records=[]
    def add(a,f,c,**tags):
        row=overlap_geometry(a,f,c)
        assert row['identity_error']<1e-12
        records.append(dict(source_table='neighbourhood_expansion',n=len(a),
            graph_sha256=hashlib.sha256(a.astype('uint8').tobytes()).hexdigest(),**tags,**row))
    for row in geometry.itertuples():
        seed=int(row.seed);g=ensemble_graph(row.kind,240,seed)
        a=nx.to_numpy_array(g,nodelist=range(240))
        perm=np.random.default_rng(seed+104729).permutation(240)
        f,c=np.sort(perm[:120]),np.sort(perm[120:])
        add(a,f,c,source_group='confirmation',kind=row.kind,graph=int(row.graph),seed=seed)
        assert np.isclose(records[-1]['clustering'],row.clustering,atol=1e-12)
        assert np.isclose(a.sum(1).std()/a.sum(1).mean(),row.degree_cv,atol=1e-12)
    archived=df[(df.source_table=='rewiring_moment_diagnostics')&(df.method=='onehop')]
    for row in archived.itertuples():
        seed=int(row.seed);g=nx.watts_strogatz_graph(240,8,float(row.p),seed=seed)
        a=nx.to_numpy_array(g,nodelist=range(240))
        perm=np.random.default_rng(seed+19).permutation(240)
        f,c=np.sort(perm[:120]),np.sort(perm[120:])
        add(a,f,c,source_group='rewiring',kind='WS_scan',graph=int(row.graph),seed=seed,rewiring_probability=float(row.p))
        assert np.isclose(records[-1]['cosine'],row.cosine,atol=1e-12)
        assert records[-1]['graph_sha256']==row.graph_sha256
    archive=ROOT/'inputs/verified_network_data.zip'
    if not archive.exists():fetch(archive)
    for info,a,f,c in load_graphs(archive):
        add(a,f,c,source_group='village',kind='Village',village=info['village'],seed=info['partition_seed'])
        assert records[-1]['graph_sha256']==info['graph_sha256']
    out=pd.DataFrame(records);assert len(out)==164
    out.to_csv(ROOT/'neighbourhood_expansion.csv',index=False)
    print(out.groupby('kind')[['expansion_ratio','edge_weight_cv_squared','cosine','uncorrected_prediction','global_expansion_ratio']].mean().to_string())
    print('Maximum identity error',out.identity_error.max())
    print('Global approximation MAE',np.mean(abs(out.global_approximation-out.cosine)))
    print('WS endpoint means:')
    print(out.query('source_group=="rewiring"').groupby('rewiring_probability')[['expansion_ratio','global_expansion_ratio','cosine']].mean().to_string())

if __name__=='__main__':run()
