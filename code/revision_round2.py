"""Assignment calibration, mechanism comparisons and pooled summaries."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from scipy import stats, sparse
import networkx as nx
import revision_study as r


def assignments(rng, reps, b, m, probability, complete):
    if not complete:
        return rng.binomial(1, probability, (reps, b+1, m)).astype(float), probability
    treated = int(np.floor(probability*m))
    keys = rng.random((reps, b+1, m))
    ix = np.argpartition(keys, treated-1, axis=2)[:, :, :treated]
    x = np.zeros_like(keys)
    np.put_along_axis(x, ix, 1., axis=2)
    assert np.all(x.sum(axis=2) == treated)
    return x, treated/m


def scores(y, k, draws, probability, complete):
    weights = np.einsum('rf,kfa->rak', y, k, optimize=True)
    if complete:
        m = weights.shape[1]
        weights = weights-weights.mean(axis=1, keepdims=True)
        sd = np.sqrt(probability*(1-probability)*m/(m-1))*np.linalg.norm(weights, axis=1)
    else:
        sd = np.sqrt(probability*(1-probability))*np.linalg.norm(weights, axis=1)
    values = np.matmul(draws-probability, weights)
    # Remove numerically constant components, including regular-graph global mode.
    return np.divide(values, sd[:, None, :], out=np.zeros_like(values),
                     where=sd[:, None, :] > 1e-12)


def tests(y, k, groups, draws, a_fc, two_fc, probability, complete, split):
    z = scores(y, k, draws, probability, complete)
    values = {name:r.rank_p(np.max(np.abs(z[:, :, ix]), axis=2))
              for name, ix in groups.items()}
    iz = np.abs(z[:, :, groups['orth_max']])
    marginal = stats.rankdata(-iz, method='max', axis=1)/iz.shape[1]
    m = iz.shape[2]
    values['bonferroni'] = np.minimum(1, m*marginal[:, 0, :].min(axis=1))
    simes = np.min(np.sort(marginal, axis=2)*m/np.arange(1, m+1), axis=2)
    cauchy = np.mean(np.tan((.5-np.clip(marginal, .5/iz.shape[1], 1-.5/iz.shape[1]))*np.pi), axis=2)
    values['simes_calibrated'] = r.rank_p(-simes)
    values['cauchy_calibrated'] = r.rank_p(cauchy)
    parts = (split[:len(split)//2], split[len(split)//2:])
    if complete:
        # Component scale does not affect validity: the entire split score is
        # exchangeably recalibrated. Do not use chi-square independence here.
        fisher = np.zeros(draws.shape[:2])
        for ix in parts:
            half = scores(y, k[groups['orth_max']][:, :, ix], draws[:, :, ix], probability, False)
            t = np.max(np.abs(half), axis=2)
            p = stats.rankdata(-t, method='max', axis=1)/t.shape[1]
            fisher += -2*np.log(p)
        values['ZZ_split_Fisher_calibrated'] = r.rank_p(fisher)
    else:
        ps = []
        for ix in parts:
            half = scores(y, k[groups['orth_max']][:, :, ix], draws[:, :, ix], probability, False)
            ps.append(r.rank_p(np.max(np.abs(half), axis=2)))
        values['ZZ_independent_CRT_Fisher'] = stats.chi2.sf(-2*np.log(np.vstack(ps)).sum(axis=0), 4)
    values['HY_COMB_fixedF'] = r.rank_p(r.hy_statistics(y, draws, a_fc))
    residual = y-y.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(residual, axis=1)[:, None]*np.sqrt(draws.sum(axis=2))
    for order, adjacency in ((1,a_fc), (2,two_fc)):
        numerator = np.einsum('rba,fa,rf->rb', draws, adjacency, residual, optimize=True)
        statistic = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator>1e-14)
        values[f'TB_graphcorr_fixedF_{order}'] = r.rank_p(np.abs(statistic))
    return values


def variance_self_test():
    import itertools
    x = np.array([a for a in itertools.product((0.,1.), repeat=5) if sum(a)==2])
    w = np.array([1.,-.7,.5,2.,-.2])
    actual = np.var(x@w)
    expected = .4*.6*5/4*np.sum((w-w.mean())**2)
    assert abs(actual-expected)<1e-12
    return float(abs(actual-expected))


def run(mode, output, quick=False):
    start = time.perf_counter()
    seed = {'null':2026100921,'complete':2026100922,'bernoulli':2026100917}[mode]
    rng = np.random.default_rng(seed)
    graphs = 1 if quick else (10 if mode=='null' else 12)
    reps = 4 if quick else (256 if mode=='null' else 32)
    b = 39 if quick else 499
    n = 80 if quick else 240
    rows = []
    for kind in r.KINDS:
        for gi in range(graphs):
            gs = int(rng.integers(0, 2**31-1))
            g, attempts = r.graph(kind, n, gs)
            a = nx.to_numpy_array(g, nodelist=range(n))
            partition = np.random.default_rng(gs+104729).permutation(n)
            f, c = np.sort(partition[:n//2]), np.sort(partition[n//2:])
            split = np.random.default_rng(gs+130363).permutation(len(c))
            w,eig,u,k,groups,orth,tails = r.prepare(a,f,c)
            cross = lambda q:q[np.ix_(f,c)]
            distance = sparse.csgraph.shortest_path(sparse.csr_matrix(a), directed=False)
            ops = {'local_W':cross(w), 'local_P':cross(r.normalized(a)[1]),
                   'local_A':cross(a), 'hard2':cross((distance<=2).astype(float)),
                   'hard3':cross((distance<=3).astype(float)),
                   'distance_decay':cross(np.exp(-(distance-1)/1.5))}
            for delta in (.1,.95):
                ops[f'offgrid_{delta}'] = cross((u*(delta*eig**2/(1-delta*eig)))@u.T)
            ops = {name:r.scale(h) for name,h in ops.items()}
            designs = (False,True) if mode=='null' else (mode=='complete',)
            for complete in designs:
                # Batch the null audit to limit permutation-array peak memory.
                for begin in range(0,reps,32):
                    count = min(32,reps-begin)
                    draws,p = assignments(rng,count,b,len(c),.4,complete)
                    x = draws[:,0,:]
                    noise = rng.normal(size=(count,len(f)))
                    mechanisms = ['null'] if mode=='null' else ['null',*ops,'threshold','independent_cascade']
                    for mechanism in mechanisms:
                        if mechanism=='null': y = noise
                        elif mechanism in ops: y = (x-p)@ops[mechanism].T+noise
                        else: y = r.cascade(a,x,c,mechanism,rng)[:,f]+noise
                        ps = tests(y,k,groups,draws,cross(a),cross((distance==2).astype(float)),p,complete,split)
                        base = ps['onehop']<=.05
                        for method,v in ps.items():
                            rejection = v<=.05
                            rows.append({'design':'complete' if complete else 'Bernoulli',
                                         'kind':kind,'graph':gi,'seed':gs,'mechanism':mechanism,
                                         'method':method,'rejects':int(rejection.sum()),
                                         'gain_sum':int((rejection.astype(int)-base).sum()),
                                         'replicates':count,'permutations':b,'n':n})
            print(mode,kind,gi+1,'/',graphs,flush=True)
    frame = pd.DataFrame(rows)
    keys = ['design','kind','graph','seed','mechanism','method','permutations','n']
    rates = frame.groupby(keys,as_index=False)[['rejects','gain_sum','replicates']].sum()
    rates['rate'] = rates.rejects/rates.replicates
    rates['paired_gain'] = rates.gain_sum/rates.replicates
    summaries = []
    for key,part in rates.groupby(['design','kind','mechanism','method']):
        mean,lo,hi = r.cluster_ci(part.rate)
        gain,glo,ghi = r.cluster_ci(part.paired_gain)
        summaries.append(dict(zip(['design','kind','mechanism','method'],key),
                              rate=mean,ci_low=max(0,lo),ci_high=min(1,hi),
                              gain=gain,gain_ci_low=glo,gain_ci_high=ghi,
                              graphs=len(part),replicates=int(part.replicates.sum())))
    pooled = []
    for key,part in rates.groupby(['design','mechanism','method']):
        mean,lo,hi = r.cluster_ci(part.rate)
        pooled.append(dict(zip(['design','mechanism','method'],key),
                           rate=mean,ci_low=max(0,lo),ci_high=min(1,hi),
                           graphs=len(part),replicates=int(part.replicates.sum())))
    output.mkdir(parents=True,exist_ok=True)
    rates.to_csv(output/f'round2_{mode}_graph_rates.csv',index=False)
    pd.DataFrame(summaries).to_csv(output/f'round2_{mode}_summary.csv',index=False)
    pd.DataFrame(pooled).to_csv(output/f'round2_{mode}_pooled.csv',index=False)
    metadata = {'seed':seed,'mode':mode,'n':n,'graphs':graphs,'reps':reps,'B':b,
                'seconds':time.perf_counter()-start,'CR_variance_selftest':variance_self_test(),
                'split':'outcome-independent random permutation',
                'complete_CRT':'jointly rank-calibrated split Fisher, not independent CRT'}
    (output/f'round2_{mode}_run.json').write_text(json.dumps(metadata,indent=2))
    print(metadata,flush=True)


def regret(output):
    path=Path(__file__).parent/'output'/'revision_graph_rates.csv'
    if path.exists():
        source=pd.read_csv(path,keep_default_na=False)
    else:
        snapshot=Path(__file__).parent.parent/'aggregate_results.csv'
        source=pd.read_csv(snapshot,keep_default_na=False,low_memory=False)
        source=source[source.source_table=='revision_graph_rates'].copy()
        source['rate']=pd.to_numeric(source['rate'],errors='raise')
    mechanisms = ['hard2','hard3','distance_decay','offgrid_0.1','offgrid_0.95','threshold','independent_cascade']
    power = source[source.mechanism.isin(mechanisms)].groupby(['method','mechanism']).rate.mean().unstack()
    # Degree-only and tail-only are diagnostics, not recommended general tests.
    candidates = power.drop(index=['degree_score','tail_W_only','tail_localspan_only'])
    ranks = candidates.rank(axis=0,ascending=False,method='average')
    gap = candidates.max(axis=0)-candidates
    result = pd.DataFrame({'mean_power':candidates.mean(axis=1),
                           'average_rank':ranks.mean(axis=1),
                           'worst_regret':gap.max(axis=1)})
    result = result.sort_values(['worst_regret','average_rank'])
    output.mkdir(parents=True,exist_ok=True)
    result.to_csv(output/'round2_posthoc_recommendation.csv')
    print(result.to_string(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['null','bernoulli','complete','regret'],required=True)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'output_round2')
    args=parser.parse_args()
    if args.mode=='regret': regret(args.output)
    else: run(args.mode,args.output,args.quick)
