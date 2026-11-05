"""Real household network topologies with prespecified simulated responses.

Only released household adjacency matrices are read. No linkage keys,
demographics, actual assignments or actual participation outcomes are used.
See VILLAGE_PROTOCOL.md for the specification written before these runs.
"""
from pathlib import Path
import io,re,zipfile,hashlib,argparse,time
import numpy as np
import pandas as pd
import networkx as nx
from network_diagnostics import response_operator,predict_threshold
from fetch_village_data import MD5,fetch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent if HERE.name=='code' else HERE
SCENARIOS=['onehop','twohop','decay']
METHODS=['onehop','matched_kernel']
THETA=np.array([0,.15,.3,.5,.75,1,1.5,2,3,4,6,8,12,16,24,40.])
SCAN=[1,4,8,12,17,21]

def load_graphs(archive):
    path=Path(archive)
    if hashlib.md5(path.read_bytes()).hexdigest()!=MD5:raise ValueError('Unexpected data archive')
    with zipfile.ZipFile(path) as z:
        pattern=re.compile(r'/adj_allVillageRelationships_HH_vilno_(\d+)\.csv$')
        members=sorted([(int(m[1]),name) for name in z.namelist() if (m:=pattern.search(name))])[:20]
        assert [v for v,_ in members]==list(range(1,13))+list(range(14,22))
        for village,name in members:
            data=z.read(name)
            matrix=np.loadtxt(io.BytesIO(data),delimiter=',')
            assert matrix.ndim==2 and matrix.shape[0]==matrix.shape[1] and np.isfinite(matrix).all()
            n0=len(matrix);loops=int(np.count_nonzero(np.diag(matrix)))
            a=((matrix>0)|(matrix.T>0)).astype(float);np.fill_diagonal(a,0)
            graph=nx.from_numpy_array(a)
            components=sorted(nx.connected_components(graph),key=lambda c:(-len(c),min(c)))
            nodes=sorted(components[0]);a=a[np.ix_(nodes,nodes)]
            seed=2026102501+104729*village
            perm=np.random.default_rng(seed).permutation(len(a))
            f,c=np.sort(perm[:len(a)//2]),np.sort(perm[len(a)//2:])
            degrees=a.sum(axis=1)
            info=dict(village=village,n_original=n0,n=len(a),edges=int(a.sum()/2),
                self_loops_removed=loops,mean_degree=degrees.mean(),degree_cv=degrees.std()/degrees.mean(),
                clustering=nx.average_clustering(nx.from_numpy_array(a)),
                source_member=name,source_sha256=hashlib.sha256(data).hexdigest(),
                graph_sha256=hashlib.sha256(a.astype('uint8').tobytes()).hexdigest(),
                partition_sha256=hashlib.sha256(np.r_[f,c].astype('<i8').tobytes()).hexdigest(),
                partition_seed=seed)
            yield info,a,f,c

def crossing(curve):
    values=np.maximum.accumulate(curve)
    eligible=np.flatnonzero(values>=.8)
    if not len(eligible):return np.inf
    j=eligible[0]
    if j==0:return float(THETA[0])
    return float(THETA[j-1]+(.8-values[j-1])*(THETA[j]-THETA[j-1])/(values[j]-values[j-1]))

def run(archive,quick=False):
    start=time.perf_counter();geometry=[];diagnostics=[];powers=[];efficiency=[]
    reps,b,boots=(16,39,40) if quick else (512,499,1000)
    for info,a,f,c in load_graphs(archive):
        village=info['village'];geometry.append(dict(source_table='village_geometry',**info))
        hs=[response_operator(a,f,c,s) for s in SCENARIOS]
        for scenario,h in zip(SCENARIOS,hs):
            for method,kernel in zip(METHODS,['onehop','matched']):
                diagnostics.append(dict(source_table='village_diagnostics',village=village,scenario=scenario,
                    method=method,**predict_threshold(a,f,c,h,kernel=kernel)))
        if village not in (SCAN[:1] if quick else SCAN):continue
        degree=a.sum(axis=1);inv=1/np.sqrt(degree)
        local=(inv[:,None]*a*inv[None,:])[np.ix_(f,c)]
        rejected=np.zeros((3,2,reps,len(THETA)),dtype=bool)
        seed=2026102502+65537*village;rng=np.random.default_rng(seed)
        for begin in range(0,reps,32):
            count=min(32,reps-begin)
            u=rng.binomial(1,.4,(count,b+1,len(c))).astype(float)-.4
            noise=rng.normal(size=(count,len(f)))
            for si,h in enumerate(hs):
                signal=u[:,0,:]@h.T
                for mi,k in enumerate([local,h]):
                    wn=noise@k;ws=signal@k
                    sn=np.einsum('rbc,rc->rb',u,wn,optimize=True)
                    ss=np.einsum('rbc,rc->rb',u,ws,optimize=True)
                    for ti,theta in enumerate(THETA):
                        score=np.abs(sn+theta*ss)
                        sd=np.sqrt(.24)*np.linalg.norm(wn+theta*ws,axis=1)
                        pvalue=(1+np.sum(score[:,1:]>=score[:,:1]-1e-12*sd[:,None],axis=1))/(b+1)
                        pvalue=np.where(sd>1e-12,pvalue,1.)
                        rejected[si,mi,begin:begin+count,ti]=pvalue<=.05
        for si,scenario in enumerate(SCENARIOS):
            for mi,method in enumerate(METHODS):
                observed=rejected[si,mi].mean(axis=0)
                for theta,rate in zip(THETA,observed):
                    powers.append(dict(source_table='village_power',village=village,scenario=scenario,method=method,
                        theta=theta,rate=rate,rejects=int(round(rate*reps)),replicates=reps,B=b,seed=seed))
                threshold=crossing(observed)
                brng=np.random.default_rng(2026102503+8191*village+101*si+mi)
                counts=brng.multinomial(reps,np.full(reps,1/reps),size=boots)
                curves=counts@rejected[si,mi]/reps
                thresholds=np.sort(np.array([crossing(row) for row in curves]))
                low=thresholds[int(np.floor(.025*(boots-1)))];high=thresholds[int(np.ceil(.975*(boots-1)))]
                prediction=next(d for d in diagnostics if d['village']==village and d['scenario']==scenario and d['method']==method)
                pred=prediction['theta']
                efficiency.append(dict(source_table='village_efficiency',village=village,scenario=scenario,method=method,
                    observed_theta80=threshold if np.isfinite(threshold) else np.nan,
                    theta80_ci_low=low if np.isfinite(low) else np.nan,theta80_ci_high=high if np.isfinite(high) else np.nan,
                    observed_status='finite' if np.isfinite(threshold) else 'right_censored',
                    bootstrap_finite=int(np.isfinite(thresholds).sum()),bootstrap_replicates=boots,
                    predicted_theta80=pred,prediction_status=prediction['status'],max_power=float(observed.max()),
                    relative_error=(pred-threshold)/threshold if pred is not None and np.isfinite(threshold) else np.nan))
        out=ROOT/('village_quick' if quick else 'village_output');out.mkdir(exist_ok=True)
        np.savez_compressed(out/f'rejection_indicators_village_{village}.npz',rejected=rejected,theta=THETA)
        print('Village',village,'scan completed, n=',len(a),'seconds=',round(time.perf_counter()-start,1),flush=True)
    out=ROOT/('village_quick' if quick else 'village_output');out.mkdir(exist_ok=True)
    for name,records in [('geometry',geometry),('diagnostics',diagnostics),('power',powers),('efficiency',efficiency)]:
        pd.DataFrame(records).to_csv(out/f'village_{name}.csv',index=False)
    e=pd.DataFrame(efficiency)
    print(e.groupby('method').agg(cases=('village','count'),observed_finite=('observed_theta80','count'),predicted_finite=('predicted_theta80','count'),MAPE=('relative_error',lambda x:x.abs().mean())).to_string(),flush=True)
    print('Elapsed seconds',round(time.perf_counter()-start,1),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=ROOT/'inputs/verified_network_data.zip')
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args()
    if not args.archive.exists():fetch(args.archive)
    run(args.archive,args.quick)
