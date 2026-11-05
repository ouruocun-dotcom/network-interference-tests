"""Sparse finite-size power calculations at fixed mean degree."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import svds
import revision_study as r

SEED=2026100923

def operators(g,f,c):
    a=nx.to_scipy_sparse_array(g,nodelist=range(len(g)),dtype=float,format='csr')
    d=np.asarray(a.sum(axis=1)).ravel()
    w=sparse.diags(1/np.sqrt(d))@a@sparse.diags(1/np.sqrt(d))
    h2=(a+a@a).tocsr();h2.data[:]=1;h2.setdiag(0);h2.eliminate_zeros()
    h3=(h2+h2@a).tocsr();h3.data[:]=1;h3.setdiag(0);h3.eliminate_zeros()
    blocks={}
    diagnostics={}
    for name,h in [('hard2',h2),('hard3',h3)]:
        block=h[f,:][:,c].tocsr()
        s=svds(block,k=1,return_singular_vectors=False,v0=np.ones(min(block.shape)),tol=1e-7)[0]
        diagnostics[name]={'pair_coverage':h.nnz/(len(g)*(len(g)-1)),
                           'FC_coverage':block.nnz/(len(f)*len(c)),
                           'singular_energy':float(s*s/block.nnz)}
        blocks[name]=block*np.sqrt(len(f)/block.nnz)
    return w.tocsr(),d,blocks,diagnostics

def coefficients(w,d,f,c,seed):
    """Projection coefficients for the controlled bank on sparse graphs.

    The adjacency inner product <Q,L> is a 32-probe randomized trace estimate;
    the principal-eigenvector inner product is exact. Both use only the graph.
    """
    local=w[f,:][:,c].tocsr()
    vf=np.sqrt(d[f])/np.sqrt(d.sum());vc=np.sqrt(d[c])/np.sqrt(d.sum())
    lg=float(vf@(local@vc));gg=float((vf@vf)*(vc@vc));ll=float(local.multiply(local).sum())
    gram=np.array([[ll,lg],[lg,gg]])
    rng=np.random.default_rng(seed+19)
    probes=rng.choice([-1.,1.],size=(len(c),32))
    rhs=np.zeros((len(d),33));rhs[c,:32]=probes;rhs[c,32]=vc
    lp=local@probes
    result=[]
    for lam in r.GRID:
        tail=w@r.solve_spd(sparse.eye(len(d),format='csr')-lam*w,rhs)-w@rhs
        lw=float(np.mean(np.sum(tail[f,:32]*lp,axis=0)))
        qg=float(vf@tail[f,32])
        result.append((lw/ll,*np.linalg.lstsq(gram,np.array([lw,qg]),rcond=1e-12)[0]))
    return local,vf,vc,result

def weights(w,d,f,c,y,coef):
    rhs=np.zeros((len(d),len(y)));rhs[f,:]=y.T
    local=np.asarray(w@rhs)[c,:].T
    global_weight=(y@(np.sqrt(d[f])/np.sqrt(d.sum())))[:,None]*(np.sqrt(d[c])/np.sqrt(d.sum()))[None,:]
    raw=[local];orth=[local];controlled=[local]
    for lam,(alpha,beta,gamma) in zip(r.GRID,coef):
        tail=np.asarray(w@r.solve_spd(sparse.eye(len(d),format='csr')-lam*w,rhs)-w@rhs)[c,:].T
        raw.append(tail);orth.append(tail-alpha*local)
        controlled.append(tail-beta*local-gamma*global_weight)
    return {'onehop':np.array([local]),'raw_max':np.array(raw),
            'orth_max':np.array(orth),'orth_degree_max':np.array(controlled)}

def evaluate(bank,draws):
    result={}
    for name,b in bank.items():
        sd=np.sqrt(.24)*np.linalg.norm(b,axis=2)
        z=np.einsum('rba,kra->rbk',draws-.4,b,optimize=True)
        z=np.divide(z,sd.T[:,None,:],out=np.zeros_like(z),where=sd.T[:,None,:]>1e-12)
        result[name]=r.rank_p(np.max(np.abs(z),axis=2))<=.05
    return result

def run(output,quick=False):
    start=time.perf_counter();rng=np.random.default_rng(SEED);records=[]
    sizes=(80,120) if quick else (240,1000,5000)
    graphs=1 if quick else 8
    reps=4 if quick else 64
    b=39 if quick else 499
    for n in sizes:
        for kind in ('ER','WS_002'):
            for gi in range(graphs):
                gs=int(rng.integers(0,2**31-1));g,_=r.graph(kind,n,gs)
                split=np.random.default_rng(gs+104729).permutation(n)
                f,c=np.sort(split[:n//2]),np.sort(split[n//2:])
                w,d,hs,diagnostics=operators(g,f,c)
                local,vf,vc,coef=coefficients(w,d,f,c,gs)
                draws=rng.binomial(1,.4,(reps,b+1,len(c))).astype(float)
                noise=rng.normal(size=(reps,len(f)))
                for name,h in hs.items():
                    y=np.asarray(h@(draws[:,0,:]-.4).T).T+noise
                    bank=weights(w,d,f,c,y,coef)
                    bank['oracle']=np.array([np.asarray(h.T@y.T).T])
                    ps=evaluate(bank,draws);baseline=ps['onehop']
                    for method,rej in ps.items():
                        records.append({'n':n,'kind':kind,'graph':gi,'seed':gs,'mechanism':name,
                                        'method':method,'rate':rej.mean(),
                                        'paired_gain':np.mean(rej.astype(int)-baseline),
                                        'replicates':reps,'B':b,'mean_degree':d.mean(),
                                        'clustering':nx.average_clustering(g),**diagnostics[name]})
                output.mkdir(parents=True,exist_ok=True)
                pd.DataFrame(records).to_csv(output/'round2_size_graph_rates.csv',index=False)
                print('size',n,kind,gi+1,'/',graphs,flush=True)
    frame=pd.DataFrame(records);rows=[]
    for key,part in frame.groupby(['n','kind','mechanism','method']):
        rate,lo,hi=r.cluster_ci(part.rate);gain,glo,ghi=r.cluster_ci(part.paired_gain)
        rows.append(dict(zip(['n','kind','mechanism','method'],key),rate=rate,
                         ci_low=max(0,lo),ci_high=min(1,hi),gain=gain,gain_ci_low=glo,
                         gain_ci_high=ghi,coverage=part.pair_coverage.mean(),
                         singular_energy=part.singular_energy.mean(),graphs=graphs,reps=reps))
    pd.DataFrame(rows).to_csv(output/'round2_size_summary.csv',index=False)
    (output/'round2_size_run.json').write_text(json.dumps({'seed':SEED,'seconds':time.perf_counter()-start,
           'sizes':sizes,'graphs':graphs,'reps':reps,'B':b,
           'projection':'fixed outcome-free 32-probe trace estimate; degree inner product exact'},indent=2))
    print('completed in',time.perf_counter()-start,'seconds',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'output_round2')
    args=p.parse_args();run(args.output,args.quick)
