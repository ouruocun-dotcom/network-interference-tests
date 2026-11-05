"""Independent confirmation across eight graph families and two assignment designs."""
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
import revision_study as r
import revision_round2 as design

CORE=('onehop','raw_max','orth_max','orth_degree_max','local_span_max')

def run(output,quick=False):
    start=time.perf_counter();rows=[];geometry=[]
    graphs=1 if quick else 12;reps=4 if quick else 64
    n=80 if quick else 240;b=39 if quick else 499
    for complete,seed in ((False,2026100931),(True,2026100932)):
        rng=np.random.default_rng(seed);label='complete' if complete else 'Bernoulli'
        for kind in r.KINDS:
            for gi in range(graphs):
                gs=int(rng.integers(0,2**31-1));g,_=r.graph(kind,n,gs)
                a=nx.to_numpy_array(g,nodelist=range(n))
                partition=np.random.default_rng(gs+104729).permutation(n)
                f,c=np.sort(partition[:n//2]),np.sort(partition[n//2:])
                w,eig,u,k,groups,orth,tails=r.prepare(a,f,c)
                cross=lambda v:v[np.ix_(f,c)]
                distance=sparse.csgraph.shortest_path(sparse.csr_matrix(a),directed=False)
                ops={'local_W':cross(w),'local_P':cross(r.normalized(a)[1]),
                     'local_A':cross(a),'hard2':cross((distance<=2).astype(float)),
                     'hard3':cross((distance<=3).astype(float)),
                     'distance_decay':cross(np.exp(-(distance-1)/1.5))}
                for delta in (.1,.95):
                    ops[f'offgrid_{delta}']=cross((u*(delta*eig**2/(1-delta*eig)))@u.T)
                ops={name:r.scale(h) for name,h in ops.items()}
                draws,p=design.assignments(rng,reps,b,len(c),.4,complete)
                noise=rng.normal(size=(reps,len(f)))
                for mechanism in ('null',*ops,'threshold','independent_cascade'):
                    if mechanism=='null':y=noise
                    elif mechanism in ops:y=(draws[:,0,:]-p)@ops[mechanism].T+noise
                    else:y=r.cascade(a,draws[:,0,:],c,mechanism,rng)[:,f]+noise
                    z=design.scores(y,k,draws,p,complete)
                    reject={name:r.rank_p(np.max(np.abs(z[:,:,groups[name]]),axis=2))<=.05 for name in CORE}
                    for name,value in reject.items():
                        rows.append({'design':label,'kind':kind,'graph':gi,'seed':gs,
                          'mechanism':mechanism,'method':name,'rate':value.mean(),
                          'gain':np.mean(value.astype(int)-reject['onehop']),
                          'rejects':int(value.sum()),'replicates':reps,'B':b,'n':n})
                geometry.append({'design':label,'kind':kind,'graph':gi,'seed':gs,
                                 'clustering':nx.average_clustering(g),'degree_cv':a.sum(1).std()/a.sum(1).mean()})
                print('confirmation',label,kind,gi+1,'/',graphs,flush=True)
    output.mkdir(parents=True,exist_ok=True);frame=pd.DataFrame(rows)
    frame.to_csv(output/'round3_confirmation_graph_rates.csv',index=False)
    pd.DataFrame(geometry).to_csv(output/'round3_confirmation_geometry.csv',index=False)
    for name,keys in (('pooled',['design','mechanism','method']),
                      ('summary',['design','kind','mechanism','method'])):
        result=[]
        for key,part in frame.groupby(keys):
            rate,lo,hi=r.cluster_ci(part.rate);gain,glo,ghi=r.cluster_ci(part.gain)
            result.append(dict(zip(keys,key),rate=rate,ci_low=max(0,lo),ci_high=min(1,hi),
                gain=gain,gain_ci_low=glo,gain_ci_high=ghi,graphs=len(part),replicates=int(part.replicates.sum())))
        pd.DataFrame(result).to_csv(output/f'round3_confirmation_{name}.csv',index=False)
    (output/'round3_confirmation_run.json').write_text(json.dumps({'seeds':[2026100931,2026100932],
        'graphs':graphs,'reps':reps,'B':b,'n':n,'seconds':time.perf_counter()-start},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'output_round3')
    args=parser.parse_args();run(args.output,args.quick)
