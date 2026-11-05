"""Two-parameter nonlinear sensitivity grid with paired signal-amplitude curves.

C seed density is .1/.4; edge transmission .1/.3 or
threshold .25/.5; observation signal amplitude 1/3/6/12, noise SD 1.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
import networkx as nx
import revision_study as r
from revision_round2 import scores

SEED=2026100925

def adoption(a,x,c,mechanism,parameter,rng):
    result=[];degree=a.sum(axis=1)
    for row in x:
        active=np.zeros(len(a),dtype=bool);active[c]=row.astype(bool)
        front=active.copy()
        for _ in range(2 if mechanism=='threshold' else 3):
            if mechanism=='threshold':
                new=(~active)&((a@active)/degree>=parameter)
            else:
                new=(~active)&(rng.random(len(a))<1-(1-parameter)**(a@front))
                front=new
            active|=new
        result.append(active)
    return np.asarray(result,dtype=float)

def run(output,quick=False):
    start=time.perf_counter();rng=np.random.default_rng(SEED);records=[]
    n=80 if quick else 240;graphs=1 if quick else 6
    reps=4 if quick else 128;b=39 if quick else 499
    for kind in ('ER','WS_002'):
        for gi in range(graphs):
            gs=int(rng.integers(0,2**31-1));g,_=r.graph(kind,n,gs)
            a=nx.to_numpy_array(g,nodelist=range(n))
            split=np.random.default_rng(gs+104729).permutation(n)
            f,c=np.sort(split[:n//2]),np.sort(split[n//2:])
            w,eig,u,k,groups,orth,tails=r.prepare(a,f,c)
            selected={name:groups[name] for name in ('onehop','raw_max','orth_max','orth_degree_max')}
            for density in (.1,.4):
                draws=rng.binomial(1,density,(reps,b+1,len(c))).astype(float)
                noise=rng.normal(size=(reps,len(f)))
                for mechanism,parameters in [('independent_cascade',(.1,.3)),('threshold',(.25,.5))]:
                    for parameter in parameters:
                        signal=adoption(a,draws[:,0,:],c,mechanism,parameter,rng)[:,f]
                        for amplitude in (1.,3.,6.,12.):
                            z=scores(amplitude*signal+noise,k,draws,density,False)
                            rr={name:r.rank_p(np.max(np.abs(z[:,:,ix]),axis=2))<=.05
                                for name,ix in selected.items()}
                            for name,rej in rr.items():
                                records.append({'kind':kind,'graph':gi,'seed':gs,'mechanism':mechanism,
                                                'density_C':density,'parameter':parameter,'amplitude':amplitude,
                                                'method':name,'rate':rej.mean(),
                                                'gain':np.mean(rej.astype(int)-rr['onehop']),
                                                'mean_adoption':signal.mean(),'reps':reps,'B':b})
            print('nonlinear',kind,gi+1,'/',graphs,flush=True)
    output.mkdir(parents=True,exist_ok=True);frame=pd.DataFrame(records)
    frame.to_csv(output/'round2_nonlinear_graph_rates.csv',index=False)
    rows=[]
    keys=['kind','mechanism','density_C','parameter','amplitude','method']
    for key,part in frame.groupby(keys):
        mean,lo,hi=r.cluster_ci(part.rate);gain,glo,ghi=r.cluster_ci(part.gain)
        rows.append(dict(zip(keys,key),rate=mean,ci_low=max(0,lo),ci_high=min(1,hi),
                         gain=gain,gain_ci_low=glo,gain_ci_high=ghi,graphs=graphs,reps=reps))
    pd.DataFrame(rows).to_csv(output/'round2_nonlinear_summary.csv',index=False)
    (output/'round2_nonlinear_run.json').write_text(json.dumps({'seed':SEED,'n':n,'graphs':graphs,
             'reps':reps,'B':b,'seconds':time.perf_counter()-start},indent=2))
    print('completed',time.perf_counter()-start,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'output_round2')
    args=p.parse_args();run(args.output,args.quick)
