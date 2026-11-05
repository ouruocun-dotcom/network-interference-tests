"""Reconstruct archived WS graphs and compute outcome-free moment diagnostics."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.csgraph import shortest_path
from scipy.stats import norm
from scipy.optimize import brentq

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent if HERE.name=='code' else HERE
df=pd.read_csv(ROOT/'aggregate_results.csv',low_memory=False)
rates=df[df.source_table=='round3_ws_rates']
eff=df[df.source_table=='round3_ws_efficiency'].dropna(axis=1,how='all')
records=[]
critical=norm.ppf(.975)
target=brentq(lambda g: norm.sf(critical-g)+norm.cdf(-critical-g)-.8,0,6)
for p in sorted(rates.p.dropna().unique()):
    for gi in range(6):
        seed=2026100933+104729*gi
        saved=rates[(rates.p==p)&(rates.graph==gi)]
        assert len(saved) and set(saved.seed.astype(int))=={seed}
        g=nx.watts_strogatz_graph(240,8,float(p),seed=seed)
        assert nx.is_connected(g)
        a=nx.to_numpy_array(g,nodelist=range(240))
        perm=np.random.default_rng(seed+19).permutation(240)
        f,c=np.sort(perm[:120]),np.sort(perm[120:])
        degree=a.sum(axis=1)
        w=a/np.sqrt(degree[:,None]*degree[None,:])
        k=w[np.ix_(f,c)]
        h=(shortest_path(a,directed=False)[np.ix_(f,c)]<=2).astype(float)
        h*=np.sqrt(120/np.sum(h*h))
        for method,kernel in [('onehop',k),('matched_kernel',h)]:
            inner=np.sum(kernel*h); kn=np.linalg.norm(kernel); hn=np.linalg.norm(h)
            product=np.linalg.norm(kernel.T@h)
            gamma=inner/product
            ratio=.24*product**2/kn**2
            weak=target/(gamma*np.sqrt(ratio))
            predicted=target/np.sqrt(ratio*(gamma**2-target**2)) if gamma>target else np.nan
            observed=eff[(eff.p==p)&(eff.graph==gi)&(eff.method==method)].theta80
            assert len(observed)==1 and np.isfinite(observed.iloc[0])
            observed=float(observed.iloc[0])
            records.append(dict(source_table='rewiring_moment_diagnostics',method=method,p=p,graph=gi,seed=seed,
                cosine=inner/(kn*hn),gamma_infinity=gamma,variance_ratio_at_theta1=ratio,
                normal_shift_for_80=target,weak_theta80=weak,moment_theta80=predicted,
                prediction_status='finite' if np.isfinite(predicted) else 'no_moment_crossing',
                observed_theta80=observed,relative_error=(predicted-observed)/observed,
                variance_ratio_at_observed_theta80=ratio*observed**2,
                graph_sha256=hashlib.sha256(a.astype('uint8').tobytes()).hexdigest(),
                partition_sha256=hashlib.sha256(np.r_[f,c].astype('<i8').tobytes()).hexdigest()))
out=pd.DataFrame(records)
out.to_csv(ROOT/'rewiring_diagnostics.csv',index=False)
print('Gaussian shift target',target)
print(out.groupby(['method','p'])[['cosine','gamma_infinity','moment_theta80','observed_theta80','relative_error']].mean().to_string())
print('Graphwise finite predictions:')
print(out.groupby(['method','p']).moment_theta80.count().to_string())
for method in ['onehop','matched_kernel']:
    q=out[(out.method==method)&(out.p<=.2)]
    print(method,'low-rewiring graphwise MAPE',q.relative_error.abs().mean(),
          'maximum absolute relative error',q.relative_error.abs().max())
z=df[(df.source_table=='round3_size_summary')&(df.kind=='ER')&(df.n==240)&(df.mechanism=='hard3')]
print('Archived finite-grid maxima:')
print(z.groupby('method').rate.max().to_string())
archived=df[(df.source_table=='round3_size_rates')&(df.kind=='ER')&(df.n==240)&(df.mechanism=='hard3')].drop_duplicates('graph')
geometry=[]
for row in archived.itertuples():
    seed=int(row.seed)
    for attempt in range(1000):
        g=nx.fast_gnp_random_graph(240,8/239,seed=seed+981*attempt)
        if nx.is_connected(g):break
    else:raise RuntimeError('Connected graph sampling exhausted')
    a=nx.to_numpy_array(g,nodelist=range(240))
    perm=np.random.default_rng(seed+104729).permutation(240)
    f,c=np.sort(perm[:120]),np.sort(perm[120:])
    h=(shortest_path(a,directed=False)[np.ix_(f,c)]<=3).astype(float)
    degree=a.sum(axis=1);v=np.sqrt(degree)/np.sqrt(degree.sum())
    principal=np.outer(v[f],v[c])
    energy=np.linalg.svd(h,compute_uv=False)[0]**2/np.sum(h*h)
    assert np.isclose(h.mean(),row.FC_coverage,atol=1e-12)
    assert np.isclose(energy,row.singular_energy,atol=1e-10)
    geometry.append(dict(source_table='er_threehop_geometry',graph=int(row.graph),seed=seed,
        FC_coverage=h.mean(),singular_energy=energy,
        principal_mode_alignment_squared=np.sum(h*principal)**2/(np.sum(h*h)*np.sum(principal*principal))))
geo=pd.DataFrame(geometry)
geo.to_csv(ROOT/'saturation_diagnostics.csv',index=False)
print('ER three-hop geometry checked against archived diagnostics:')
print(geo[['FC_coverage','singular_energy','principal_mode_alignment_squared']].mean().to_string())
