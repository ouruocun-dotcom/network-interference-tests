"""Independent WS extension and Bernoulli size-efficiency curves."""
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
import revision_study as r
import revision_size as size
import revision_controlled as controlled

BASE=np.array((.05,.1,.2,.35,.5,.75,1.,1.5,2.,3.,5.,8.,12.,20.,40.))

def interpolate(x,y):
    y=np.maximum.accumulate(y)
    if y[0]>=.8:return np.nan,'left'
    if y[-1]<.8:return np.nan,'right'
    j=int(np.searchsorted(y,.8))
    return float(x[j-1]+(.8-y[j-1])*(x[j]-x[j-1])/(y[j]-y[j-1])),'none'

def summarize(frame,output,prefix,cellkeys):
    summaries=[];eff=[]
    for key,part in frame.groupby(cellkeys+['theta','method']):
        rate,lo,hi=r.cluster_ci(part.rate)
        summaries.append(dict(zip(cellkeys+['theta','method'],key),rate=rate,ci_low=max(0,lo),ci_high=min(1,hi)))
    for key,part in frame.groupby(cellkeys+['graph']):
        curve=part.pivot(index='theta',columns='method',values='rate').sort_index()
        values={method:interpolate(curve.index.to_numpy(),curve[method].to_numpy()) for method in curve}
        for method,(theta,censor) in values.items():
            eff.append(dict(zip(cellkeys+['graph'],key),method=method,theta80=theta,censor=censor,
               matched_relative_strength=theta/values['matched_kernel'][0],
               onehop_relative_efficiency=values['onehop'][0]/theta))
    pd.DataFrame(summaries).to_csv(output/f'{prefix}_summary.csv',index=False)
    efficiency=pd.DataFrame(eff);efficiency.to_csv(output/f'{prefix}_efficiency.csv',index=False)
    intervals=[]
    for key,part in efficiency.groupby(cellkeys+['method']):
        for quantity in ('theta80','matched_relative_strength','onehop_relative_efficiency'):
            v=part[quantity].dropna().to_numpy()
            if len(v):mean,lo,hi=r.cluster_ci(v)
            else:mean=lo=hi=np.nan
            intervals.append(dict(zip(cellkeys+['method'],key),quantity=quantity,mean=mean,ci_low=lo,ci_high=hi,
                       observed=len(v),total=len(part),right=int((part.censor=='right').sum()),left=int((part.censor=='left').sum())))
    pd.DataFrame(intervals).to_csv(output/f'{prefix}_intervals.csv',index=False)

def run_ws(output,quick=False):
    controlled.SEED=2026100933
    controlled.THETA=(.15,.3,.5,.75,1.,1.5,2.,3.,4.,6.,8.,10.,12.,16.)
    controlled.run(output,quick)
    # Reuse the tested kernel/draw engine, then explicitly relabel the matched
    # comparator. The old module and its original outputs remain reproducible.
    path=output/'round2_controlled_rates.csv'
    frame=pd.read_csv(path);frame['method']=frame.method.replace({'oracle':'matched_kernel'})
    frame.to_csv(output/'round3_ws_rates.csv',index=False)
    summarize(frame,output,'round3_ws',['p'])

def run_size(output,quick=False):
    start=time.perf_counter();rng=np.random.default_rng(2026100934);records=[]
    graphs=1 if quick else 8;reps=4 if quick else 64;b=39 if quick else 499
    sizes=(80,120) if quick else (240,1000,5000)
    output.mkdir(parents=True,exist_ok=True)
    for n in sizes:
        theta_grid=BASE*np.sqrt(240/n)
        for kind in ('ER','WS_002'):
            for gi in range(graphs):
                gs=int(rng.integers(0,2**31-1));g,_=r.graph(kind,n,gs)
                split=np.random.default_rng(gs+104729).permutation(n)
                f,c=np.sort(split[:n//2]),np.sort(split[n//2:])
                w,d,hs,diagnostics=size.operators(g,f,c)
                _,_,_,coef=size.coefficients(w,d,f,c,gs)
                # Batches limit peak assignment memory at 5,000 vertices.
                counts={}
                for begin in range(0,reps,16):
                    count=min(16,reps-begin)
                    draws=rng.binomial(1,.4,(count,b+1,len(c))).astype(float)
                    noise=rng.normal(size=(count,len(f)))
                    nw=size.weights(w,d,f,c,noise,coef)
                    for mechanism,h in hs.items():
                        signal=np.asarray(h@(draws[:,0,:]-.4).T).T
                        sw=size.weights(w,d,f,c,signal,coef)
                        nn=dict(nw);ss=dict(sw)
                        nn['matched_kernel']=np.array([np.asarray(h.T@noise.T).T])
                        ss['matched_kernel']=np.array([np.asarray(h.T@signal.T).T])
                        for theta in theta_grid:
                            rejected=size.evaluate({name:nn[name]+theta*ss[name] for name in nn},draws)
                            for name,value in rejected.items():
                                key=(mechanism,float(theta),name)
                                counts[key]=counts.get(key,0)+int(value.sum())
                for (mechanism,theta,method),count in counts.items():
                    records.append({'n':n,'kind':kind,'graph':gi,'seed':gs,'mechanism':mechanism,
                     'theta':theta,'method':method,'rate':count/reps,'rejects':count,
                     'replicates':reps,'B':b,**diagnostics[mechanism]})
                pd.DataFrame(records).to_csv(output/'round3_size_rates.csv',index=False)
                print('size-efficiency',n,kind,gi+1,'/',graphs,flush=True)
    summarize(pd.DataFrame(records),output,'round3_size',['n','kind','mechanism'])
    (output/'round3_size_run.json').write_text(json.dumps({'seed':2026100934,'sizes':sizes,
       'graphs':graphs,'reps':reps,'B':b,'base_grid':BASE.tolist(),'scale':'sqrt(240/n)',
       'seconds':time.perf_counter()-start},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['ws','size'],required=True)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'output_round3')
    args=parser.parse_args()
    (run_ws if args.mode=='ws' else run_size)(args.output,args.quick)
