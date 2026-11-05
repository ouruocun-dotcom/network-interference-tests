"""Network ensembles, response mechanisms and common-draw test comparisons."""
from __future__ import annotations
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats, sparse

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
GRID = (.2, .4, .6, .75, .85, .92)
KINDS = ('ER', 'Regular', 'BA', 'SBM_003', 'SBM_045', 'WS_002', 'WS_020', 'WS_100')
SEED = 2026100917

def graph(kind, n, seed):
    for attempt in range(1000):
        s = seed + 981 * attempt
        if kind == 'ER':
            g = nx.fast_gnp_random_graph(n, 8/(n-1), seed=s)
        elif kind == 'Regular':
            g = nx.random_regular_graph(8, n, seed=s)
        elif kind == 'BA':
            g = nx.barabasi_albert_graph(n, 4, seed=s)
        elif kind.startswith('SBM'):
            mix = .03 if kind.endswith('003') else .45
            q = n//2
            g = nx.stochastic_block_model([q, n-q],
                [[8*(1-mix)/(q-1), 8*mix/q], [8*mix/q, 8*(1-mix)/(q-1)]], seed=s)
        else:
            rewiring = {'WS_002': .02, 'WS_020': .2, 'WS_100': 1.0}[kind]
            g = nx.watts_strogatz_graph(n, 8, rewiring, seed=s)
        if nx.is_connected(g):
            return g, attempt + 1
    raise RuntimeError('Connected graph sampling exhausted')

def normalized(a):
    d = a.sum(axis=1)
    inv = np.divide(1, np.sqrt(d), out=np.zeros_like(d), where=d>0)
    w = inv[:,None]*a*inv[None,:]
    p = np.divide(a, d[:,None], out=np.zeros_like(a), where=d[:,None]>0)
    v = np.sqrt(d)/np.sqrt(d.sum())
    return w, p, np.outer(v,v)

def project(q, bases):
    x = np.column_stack([b.ravel() for b in bases])
    # SVD least squares automatically handles A/P/W collinearity on regular graphs.
    coefficients = np.linalg.lstsq(x, q.ravel(), rcond=1e-10)[0]
    return q - (x @ coefficients).reshape(q.shape)

def scale(h):
    norm = np.linalg.norm(h)
    return h * np.sqrt(h.shape[0])/norm if norm > 1e-14 else h.copy()

def prepare(a, f, c):
    w, p, degree = normalized(a)
    eig, u = np.linalg.eigh(w)
    cross = lambda x: x[np.ix_(f,c)]
    local, row, raw, central = map(cross, (w,p,a,degree))
    tails = [cross((u*(lam*eig**2/(1-lam*eig))) @ u.T) for lam in GRID]
    orth = [project(q,[local]) for q in tails]
    wg = [project(q,[local,central]) for q in tails]
    span = [project(q,[local,row,raw,central]) for q in tails]
    names = ['W','P','A','degree'] + [f'raw_{x}' for x in GRID] + [f'orth_{x}' for x in GRID] + [f'wg_{x}' for x in GRID] + [f'span_{x}' for x in GRID] + ['Katz_06']
    allk = np.array([local,row,raw,central,*tails,*orth,*wg,*span,local+tails[2]])
    ids = {name:i for i,name in enumerate(names)}
    bank = lambda prefix: [ids['W']]+[ids[f'{prefix}_{x}'] for x in GRID]
    groups = {'onehop':[0], 'raw_max':bank('raw'), 'orth_max':bank('orth'),
              'degree_score':[3], 'katz_single':[ids['Katz_06']],
              'orth_degree_max':bank('wg'),
              'local_span_max':[0,1,2]+[ids[f'span_{x}'] for x in GRID],
              'small_grid_max':[0]+[ids[f'orth_{x}'] for x in (.2,.6,.85)],
              'tail_W_only':[ids[f'orth_{x}'] for x in GRID],
              'tail_localspan_only':[ids[f'span_{x}'] for x in GRID]}
    return w, eig, u, allk, groups, orth, tails

def rank_p(t):
    return (np.sum(t[:,1:] >= t[:,0,None]-1e-12, axis=1)+1)/t.shape[1]

def zscores(y, k, draws, probability=.4):
    weights = np.einsum('rf,kfa->rak',y,k,optimize=True)
    denominator = np.sqrt(probability*(1-probability))*np.linalg.norm(weights,axis=1)
    values = np.matmul(draws-probability, weights)
    return np.divide(values, denominator[:,None,:], out=np.zeros_like(values),
                     where=denominator[:,None,:]>1e-14)

def hy_statistics(y, draws, adjacency_cross):
    """Hoshino/Yanagi section 3.3 COMB, binary any-treated-neighbour exposure.

    Fixed F treatment=0; E0=own treatment and E1=(own, any treated neighbour).
    Empty exposure groups contribute zero, fixed for every reassignment.
    """
    exposed = np.matmul(draws, adjacency_cross.T)>0
    n = y.shape[1]
    n1 = exposed.sum(axis=2)
    n0 = n-n1
    valid = (n1>0)&(n0>0)
    s1 = np.einsum('rbf,rf->rb',exposed,y,optimize=True)
    s0 = y.sum(axis=1)[:,None]-s1
    difference = s1/np.maximum(n1,1)-s0/np.maximum(n0,1)
    ranks = stats.rankdata(y,axis=1)
    r1 = np.einsum('rbf,rf->rb',exposed,ranks,optimize=True)
    r0 = ranks.sum(axis=1)[:,None]-r1
    rd = r1/np.maximum(n1,1)-r0/np.maximum(n0,1)
    kw = 12/(n*(n+1)) * n0*n1/n * rd**2
    between = n0*n1/n * difference**2
    total = np.sum((y-y.mean(axis=1)[:,None])**2,axis=1)[:,None]
    ols = (n-2)*between/np.maximum(total-between,1e-12)
    return np.where(valid,kw+np.abs(difference)+ols,0.0)

def test_bank(y,k,groups,draws,a_fc,distance_two_fc):
    z = zscores(y,k,draws)
    values = {name:rank_p(np.max(np.abs(z[:,:,indices]),axis=2))
              for name,indices in groups.items()}
    iz = np.abs(z[:,:,groups['orth_max']])
    m = iz.shape[2]
    # Marginal augmented ranks are permutation-equivariant; recalibrate the
    # entire Simes/Cauchy statistic, not the selected observed p-values alone.
    marginal = stats.rankdata(-iz,method='max',axis=1)/iz.shape[1]
    values['bonferroni'] = np.minimum(1,m*np.min(marginal[:,0,:],axis=1))
    simes = np.min(np.sort(marginal,axis=2)*m/np.arange(1,m+1),axis=2)
    cct = np.mean(np.tan((.5-np.clip(marginal, .5/iz.shape[1],1-.5/iz.shape[1]))*np.pi),axis=2)
    values['simes_calibrated'] = rank_p(-simes)
    values['cauchy_calibrated'] = rank_p(cct)
    split = draws.shape[2]//2
    ps = []
    for sl in (slice(None,split),slice(split,None)):
        zs = zscores(y,k[groups['orth_max'],:,sl],draws[:,:,sl])
        ps.append(rank_p(np.max(np.abs(zs),axis=2)))
    values['ZZ_independent_CRT_Fisher'] = stats.chi2.sf(-2*np.log(np.vstack(ps)).sum(axis=0),4)
    values['HY_COMB_fixedF'] = rank_p(hy_statistics(y,draws,a_fc))
    # Tiwari/Basu eqs. (12)-(13), restricted to the common fixed-F null.
    # Own F treatment is always zero: intercept-only residual refitting leaves
    # Y_F minus its mean unchanged. This is not their general fitted-model test.
    residual = y-y.mean(axis=1,keepdims=True)
    denominator = np.linalg.norm(residual,axis=1)[:,None]*np.sqrt(draws.sum(axis=2))
    for order,adjacency in ((1,a_fc),(2,distance_two_fc)):
        numerator = np.einsum('rba,fa,rf->rb',draws,adjacency,residual,optimize=True)
        statistic = np.divide(numerator,denominator,out=np.zeros_like(numerator),
                              where=denominator>1e-14)
        values[f'TB_graphcorr_fixedF_{order}'] = rank_p(np.abs(statistic))
    return values, z

def cascade(a,assignments,c,mechanism,rng):
    outputs = []
    for x in assignments:
        adopted = np.zeros(len(a),dtype=bool)
        adopted[c] = x.astype(bool)
        infectious = adopted.copy()
        for _ in range(2 if mechanism=='threshold' else 3):
            if mechanism=='threshold':
                new = (~adopted)&((a@adopted)/a.sum(axis=1)>=.5)
            else:
                counts = a@infectious
                new = (~adopted)&(rng.random(len(a)) < 1-(1-.12)**counts)
                infectious = new
            adopted |= new
        outputs.append(adopted)
    return np.array(outputs,dtype=float)

def cluster_ci(x):
    x = np.asarray(x)
    mean = float(x.mean())
    half = stats.t.ppf(.975,len(x)-1)*np.std(x,ddof=1)/np.sqrt(len(x)) if len(x)>1 else np.nan
    return mean,mean-half,mean+half

def moment_proxy(h,k,p=.4,noise=1):
    variance_x = p*(1-p)
    means = variance_x*np.einsum('kfa,fa->k',k,h)
    expected_v = variance_x*(noise**2*np.sum(k*k,axis=(1,2)) +
                     variance_x*np.sum(np.matmul(np.swapaxes(k,1,2),h)**2,axis=(1,2)))
    return np.divide(means,np.sqrt(expected_v),out=np.zeros_like(means),where=expected_v>1e-14)

def gaussian_proxy(h,k,groups,criticals,rng,size=20000):
    """Moment-matched working approximation, not a power theorem."""
    kh = np.matmul(np.swapaxes(k,1,2),h)
    covariance = .24*(np.einsum('kfa,lfa->kl',k,k)+.24*np.einsum('kab,lab->kl',kh,kh))
    den = np.sqrt(np.diag(covariance))
    correlation = np.divide(covariance,np.outer(den,den),out=np.zeros_like(covariance),
                            where=np.outer(den,den)>1e-14)
    ev,vec = np.linalg.eigh(correlation)
    gamma = moment_proxy(h,k)
    normal = (rng.normal(size=(size,len(k)))*np.sqrt(np.maximum(ev,0)))@vec.T+gamma
    return {method:np.mean(np.max(np.abs(normal[:,indices]),axis=1)>criticals[method])
            for method,indices in groups.items()},gamma

def moment_self_test():
    import itertools
    h=np.array([[1.,2.,-.3],[.4,-1.,.7]])
    k=np.array([[.2,1.,-.5],[1.,.3,.4]])
    design=np.array(list(itertools.product((0.,1.),repeat=3)))
    probability=.4
    mass=np.prod(np.where(design==1,probability,1-probability),axis=1)
    u=design-probability
    signal=u@h.T
    score=np.einsum('rf,fa,ra->r',signal,k,u)
    expected_mean=.24*np.sum(h*k)
    expected_v=.24*(np.sum(k*k)+.24*np.sum((k.T@h)**2))
    enum_v=np.sum(mass*.24*np.sum((signal@k)**2,axis=1))+.24*np.sum(k*k)
    assert abs(np.sum(mass*score)-expected_mean)<1e-12
    assert abs(enum_v-expected_v)<1e-12
    return {'mean_error':float(abs(np.sum(mass*score)-expected_mean)),
            'variance_error':float(abs(enum_v-expected_v))}

def geometry_self_test():
    a=np.array([[0,1,1,0,0,1],[1,0,1,1,0,0],[1,1,0,1,1,0],
                [0,1,1,0,1,0],[0,0,1,1,0,1],[1,0,0,0,1,0]],dtype=float)
    w,_,degree=normalized(a)
    f=np.array([0,2,4]);c=np.array([1,3,5])
    cross=lambda x:x[np.ix_(f,c)]
    triangle=np.einsum('ij,ik,kj->',w*np.outer(np.isin(range(6),f),np.isin(range(6),c)),w,w)
    triangle_error=abs(np.sum(cross(w@w)*cross(w))-triangle)
    eig,u=np.linalg.eigh(w)
    gv=eig+eig**2;hv=eig**3
    g=(u*gv)@u.T;h=(u*hv)@u.T
    mf=u[f].T@u[f];mc=u[c].T@u[c]
    mask_error=abs(np.sum(cross(g)*cross(h))-np.trace(np.diag(gv)@mf@np.diag(hv)@mc))
    bank=np.array([cross(degree)])
    draws=np.array([[[0.,1.,0.],[1.,0.,1.],[1.,1.,0.]]])
    z1=np.abs(zscores(np.array([[1.,2.,3.]]),bank,draws))
    z2=np.abs(zscores(np.array([[-2.,1.,-.5]]),bank,draws))
    rankone_error=np.max(np.abs(z1-z2))
    residual=project(cross(g),[cross(w)])
    projection_error=abs(np.sum(residual*cross(w)))
    checks={'triangle_error':float(triangle_error),'mask_error':float(mask_error),
            'rankone_error':float(rankone_error),'projection_error':float(projection_error)}
    assert max(checks.values())<1e-12,checks
    return checks

def run(n=240,graphs=12,reps=32,permutations=499):
    start = time.perf_counter()
    rng = np.random.default_rng(SEED)
    proxy_rng = np.random.default_rng(SEED+91)
    identity = moment_self_test()
    rows,structural,components,proxies = [],[],[],[]
    for kind in KINDS:
        for gi in range(graphs):
            seed = int(rng.integers(0,2**31-1))
            g, attempts = graph(kind,n,seed)
            a = nx.to_numpy_array(g,nodelist=range(n))
            split = np.random.default_rng(seed+104729).permutation(n)
            f,c = np.sort(split[:n//2]),np.sort(split[n//2:])
            w,eig,u,k,groups,orth,tails = prepare(a,f,c)
            distances = np.asarray(sparse.csgraph.shortest_path(sparse.csr_matrix(a),directed=False))
            cross = lambda q:q[np.ix_(f,c)]
            operators = {'local_W':cross(w),'local_P':cross(normalized(a)[1]),
                         'local_A':cross(a), 'hard2':cross((distances<=2).astype(float)),
                         'hard3':cross((distances<=3).astype(float)),
                         'distance_decay':cross(np.exp(-(distances-1)/1.5))}
            for delta in (.1,.95):
                operators[f'offgrid_{delta}'] = cross((u*(delta*eig**2/(1-delta*eig)))@u.T)
            operators = {name:scale(h) for name,h in operators.items()}
            draws = rng.binomial(1,.4,(reps,permutations+1,len(c))).astype(float)
            x = draws[:,0,:]
            noise = rng.normal(size=(reps,len(f)))
            d = a.sum(axis=1)
            partition = [set(range(n//2)),set(range(n//2,n))] if kind.startswith('SBM') else nx.community.greedy_modularity_communities(g)
            metrics = {'kind':kind,'graph':gi,'seed':seed,'attempts':attempts,
                       'nodes':n,'mean_degree':d.mean(),'degree_cv':d.std(ddof=1)/d.mean(),
                       'clustering':nx.average_clustering(g),'modularity':nx.community.modularity(g,partition),
                       'spectral_gap':1-eig[-2], 'principal_ipr':np.sum((np.sqrt(d)/np.sqrt(d.sum()))**4)}
            local = cross(w)
            for lam,q,r in zip(GRID,tails,orth):
                share = np.sum(q*local)**2/(np.sum(q*q)*np.sum(local*local))
                degree_share = np.sum(q*cross(normalized(a)[2]))**2/(np.sum(q*q)*np.sum(cross(normalized(a)[2])**2))
                structural.append({**metrics,'lambda':lam,'projection_share_W':share,
                                    'alignment_degree_squared':degree_share})
            for mechanism in ['null',*operators,'threshold','independent_cascade']:
                h = operators.get(mechanism)
                if mechanism=='null': y = noise.copy()
                elif h is not None: y = (x-.4)@h.T+noise
                else: y = cascade(a,x,c,mechanism,rng)[:,f]+noise
                pvals,z = test_bank(y,k,groups,draws,cross(a),cross((distances==2).astype(float)))
                for method,ps in pvals.items():
                    reject = ps<=.05
                    baseline = pvals['onehop']<=.05
                    gain = reject.astype(float)-baseline
                    rows.append({'kind':kind,'graph':gi,'seed':seed,'mechanism':mechanism,
                                 'method':method,'rate':reject.mean(),'paired_gain':gain.mean(),
                                 'replicates':reps,'permutations':permutations})
                if mechanism=='null':
                    # Conditional score correlation, averaged over independent Y.
                    corr = []
                    for yy in y:
                        weights = k[groups['orth_max']].transpose(0,2,1)@yy
                        den = np.linalg.norm(weights,axis=1)
                        corr.append((weights@weights.T)/np.outer(den,den))
                    average = np.mean(corr,axis=0)
                    effective = 7**2/np.sum(average**2)
                    criticals = {method:np.quantile(np.max(np.abs(z[:,1:,indices]),axis=2),.95)
                                 for method,indices in groups.items()}
                    critical = criticals['orth_max']
                    for i in range(7):
                        for j in range(7):
                            components.append({**metrics,'component_i':i,'component_j':j,
                                                'correlation':average[i,j],'effective_rank':effective,
                                                'critical_value':critical})
                if h is not None:
                    predicted,gamma = gaussian_proxy(h,k,groups,criticals,proxy_rng)
                    for method in ('onehop','orth_max','raw_max','orth_degree_max','local_span_max'):
                        proxies.append({**metrics,'mechanism':mechanism,'method':method,
                                        'max_gamma':np.max(np.abs(gamma[groups[method]])),
                                        'power_proxy':predicted[method],'proxy_draws':20000,
                                        'observed_rate':(pvals[method]<=.05).mean()})
            print(kind,gi+1,'/',graphs,flush=True)
    frame = pd.DataFrame(rows)
    summaries = []
    for key,part in frame.groupby(['kind','mechanism','method'],sort=False):
        rate,lo,hi = cluster_ci(part.rate)
        gain,glo,ghi = cluster_ci(part.paired_gain)
        summaries.append(dict(zip(['kind','mechanism','method'],key),rate=rate,
                         ci_low=max(0,lo),ci_high=min(1,hi),paired_gain=gain,
                         gain_ci_low=glo,gain_ci_high=ghi,graphs=graphs,replicates_per_graph=reps,
                         permutations=permutations,n=n))
    OUT.mkdir(parents=True,exist_ok=True)
    frame.to_csv(OUT/'revision_graph_rates.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'revision_summary.csv',index=False)
    pd.DataFrame(structural).to_csv(OUT/'revision_topology.csv',index=False)
    pd.DataFrame(components).to_csv(OUT/'revision_correlations.csv',index=False)
    pd.DataFrame(proxies).to_csv(OUT/'revision_proxy.csv',index=False)
    metadata = {'seed':SEED,'n':n,'graphs_per_kind':graphs,'replicates_per_graph':reps,
                'permutations':permutations,'seconds':time.perf_counter()-start,
                'python':platform.python_version(),'numpy':np.__version__,'direction':'two-sided',
                'moment_identity_selftest':identity}
    (OUT/'revision_run.json').write_text(json.dumps(metadata,indent=2))
    print(metadata,flush=True)

def solve_spd(matrix, rhs, tolerance=1e-8):
    """Independent conjugate gradients batched by RHS; explicit residual audit."""
    rhs = np.asarray(rhs)
    one = rhs.ndim==1
    b = rhs[:,None] if one else rhs
    x = np.zeros_like(b)
    residual = b.copy()
    direction = residual.copy()
    squared = np.sum(residual*residual,axis=0)
    initial = squared.copy()
    for iteration in range(500):
        active = squared > tolerance**2*np.maximum(initial,1e-30)
        if not active.any(): break
        product = matrix@direction
        curvature = np.sum(direction*product,axis=0)
        alpha = np.divide(squared,curvature,out=np.zeros_like(squared),where=active)
        x += direction*alpha
        residual -= product*alpha
        next_squared = np.sum(residual*residual,axis=0)
        beta = np.divide(next_squared,squared,out=np.zeros_like(squared),where=active)
        direction = residual+direction*beta
        direction[:,~active] = 0
        squared = next_squared
    relative = np.linalg.norm(matrix@x-b,axis=0)/np.maximum(np.linalg.norm(b,axis=0),1e-30)
    if np.max(relative)>2*tolerance:
        raise RuntimeError(f'CG failed residual audit: {np.max(relative)}')
    return x[:,0] if one else x

def sparse_weights(a,f,c,y):
    d = np.asarray(a.sum(axis=1)).ravel()
    inv = np.divide(1,np.sqrt(d),out=np.zeros_like(d),where=d>0)
    w = sparse.diags(inv)@a@sparse.diags(inv)
    local = w[f,:][:,c].tocsr()
    yy = np.zeros(len(d)); yy[f] = y
    local_weight = np.asarray(w@yy)[c]
    denom = float(local.multiply(local).sum())
    weights = [local_weight]
    # Exact projection coefficients only require entries on cross edges, not
    # a dense FC matrix. Small networks use batched columns; large networks
    # estimate this trace with fixed graph-only Rademacher probes. Such estimates
    # are outcome-free, hence preserve randomization validity, but are labelled
    # approximate and verified separately against the dense small graph.
    rng = np.random.default_rng(SEED+1)
    probes = rng.choice([-1.,1.],size=(len(c),32))
    full = np.zeros((len(d),32));full[c,:] = probes
    local_probes = local@probes
    coefficients = []
    for lam in GRID:
        matrix = (sparse.eye(len(d),format='csr')-lam*w).tocsr()
        solved = solve_spd(matrix,yy)
        tail_weight = np.asarray(w@solved-w@yy)[c]
        tail_probes = np.asarray(w@solve_spd(matrix,full)-w@full)[f,:]
        coefficient = float(np.mean(np.sum(tail_probes*local_probes,axis=0))/denom)
        weights.append(tail_weight-coefficient*local_weight)
        coefficients.append(coefficient)
    return np.column_stack(weights), coefficients

def scaling():
    records = []
    rng = np.random.default_rng(SEED+2)
    for n in (240,10000,100000):
        begin = time.perf_counter()
        g = nx.fast_gnp_random_graph(n,8/(n-1),seed=SEED+n)
        a = nx.to_scipy_sparse_array(g,nodelist=range(n),dtype=float,format='csr')
        split = rng.permutation(n);f,c = np.sort(split[:n//2]),np.sort(split[n//2:])
        y = rng.normal(size=len(f))
        built = time.perf_counter()
        weights,coef = sparse_weights(a,f,c,y)
        solved = time.perf_counter()
        denominator = np.sqrt(.24)*np.linalg.norm(weights,axis=0)
        observed = rng.binomial(1,.4,len(c))
        zs = (observed-.4)@weights/denominator
        exceed=0
        for start in range(0,499,25):
            count = min(25,499-start)
            zx = (rng.binomial(1,.4,(count,len(c)))-.4)@weights/denominator
            exceed += np.sum(np.max(np.abs(zx),axis=1)>=np.max(np.abs(zs)))
        finished = time.perf_counter()
        error = np.nan
        if n==240:
            _,_,_,k,groups,_,_ = prepare(a.toarray(),f,c)
            # Fixed approximate coefficients must match dense evaluation of the
            # same approximate projection, not silently claim exact projection.
            w,_,_ = normalized(a.toarray())
            dense = [w[np.ix_(f,c)].T@y]
            for lam,cc in zip(GRID,coef):
                q = w@np.linalg.solve(np.eye(n)-lam*w,np.eye(n))-w
                dense.append((q[np.ix_(f,c)]-cc*w[np.ix_(f,c)]).T@y)
            error = np.max(np.abs(weights-np.column_stack(dense)))
            assert error < 1e-7
        records.append({'n':n,'edges':g.number_of_edges(),'graph_seconds':built-begin,
                        'kernel_seconds':solved-built,'score_seconds':finished-solved,
                        'pvalue':(1+exceed)/500,'probes':32,'sparse_dense_error':error,
                        'projection':'graph-only randomized trace estimate'})
        pd.DataFrame(records).to_csv(OUT/'revision_scaling.csv',index=False)
        print(records[-1],flush=True)

if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--scaling',action='store_true')
    p.add_argument('--graphs',type=int,default=12)
    p.add_argument('--output',type=Path)
    args = p.parse_args()
    if args.output: OUT=args.output.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.scaling: scaling()
    else: run(n=80 if args.quick else 240,graphs=1 if args.quick else args.graphs,
              reps=4 if args.quick else 32,permutations=39 if args.quick else 499)
