"""Prepare manuscript figures and tables from archived results; no simulations.

Run from the manuscript directory, or from a public repository containing
aggregate_results.csv and code/build_assets.py. Reference inputs are not edited.
"""
from pathlib import Path
import argparse, re, shutil, json, hashlib, zipfile
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent if HERE.name=='code' else HERE
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,default=ROOT/'aggregate_results.csv')
args=parser.parse_args()
ROOT.joinpath('tables').mkdir(exist_ok=True)
ROOT.joinpath('figures').mkdir(exist_ok=True)
df=pd.read_csv(args.source,low_memory=False)
def source(name):
    x=df.loc[df.source_table==name].copy()
    assert len(x), name
    return x.dropna(axis=1,how='all')
def f(x): return str(Decimal(str(float(x))).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
def ci(a,b): return '['+f(a)+', '+f(b)+']'
CAPTIONS = {'village_thresholds.tex': 'Graph-only predictions (Pred.) and observed thresholds $\\theta_{80}$ (Obs.) on six village networks; 512 simulated datasets per village and response, 499 re-randomizations, 16 signal strengths up to 40. For the one-hop response the one-hop and matched tests coincide, so one value is shown. For two-hop and decay responses no graph-only prediction exists; Obs.~1 and Obs.~M give the one-hop and matched tests, and $>40$ means power never reached 0.8 on the grid.', 'calibration.tex': 'False-positive rates (\\%) at level 0.05 when there is no spillover: 20,480 null datasets on 80 graphs per design, 499 re-randomizations each. Intervals are 95\\% intervals across graphs.', 'village_geometry.tex': 'The six scanned village networks: households before ($n_0$) and after ($n$) keeping the largest connected component, mean degree and clustering coefficient. Profiles of all 20 villages are in the reference CSV.', 'mechanisms.tex': 'Power at unit signal strength in the independent confirmation (level 0.05). Each entry pools 6,144 simulated datasets on 96 graphs, 12 from each of eight families, with 499 re-randomizations per test. Responses: 2-hop and 3-hop reach every pair within two or three steps; D decays exponentially with distance; $\\delta=.1$ and $\\delta=.95$ are Katz responses with parameters outside the scan grid; T is threshold adoption and IC independent cascade. The default rule uses the controlled bank under Bernoulli and the raw bank under fixed-count assignment. False-positive rates are in Supplementary Table~S6.', 'paired.tex': 'Power of the recommended bank, pooled over all eight graph families, and its gain over the one-hop test in percentage points (pp). B: Bernoulli assignment, controlled bank; F: fixed-count assignment, raw bank. Intervals are 95\\% intervals across graphs.', 'confirmation_full.tex': 'Power at unit signal strength in the independent confirmation, by graph family and response (12 graphs and 64 datasets per graph; level 0.05). Columns 1, R and C give power for the one-hop test and the raw and controlled banks. Gain is the improvement of the recommended bank (C under Bernoulli, R under fixed counts) over one-hop, in percentage points, with a 95\\% interval across graphs. Response labels as in main Table~1.', 'threshold_predictions.tex': 'Graph-only diagnostics and threshold predictions on the 48 WS rewiring graphs (six per rewiring level). Score 1: one-hop test; M: matched test. Cosine is the overlap with the response, $\\Gamma_\\infty$ the saturation level and $r$ the signal-to-noise variance ratio at unit strength, each averaged over six graphs. Finite: number of graphs with a prediction ($\\Gamma_\\infty>2.80$). Observed: mean observed $\\theta_{80}$ (all finite). Weak: weak-signal prediction. Pred.: prediction from main Eq.~(9). MAPE: mean absolute percentage error of Pred. against the observed threshold on the same graph. Pred. and MAPE average only graphs with a prediction, so at rewiring 0.5 the one-hop values rest on two graphs.'}
def save(name,s):
    """Write a table source file, inserting the manuscript caption."""
    start=s.index('\\caption{')+len('\\caption{')
    end=s.index('}\\label{',start)
    s=s[:start]+CAPTIONS[name]+s[end:]
    (ROOT/'tables'/name).write_text(s+'\n')
def tab(name,label,cols,rows):
    s='\\begin{table}[htbp]\n\\centering\\small\\setlength{\\tabcolsep}{3pt}\n'
    s+='\\caption{}\\label{'+label+'}\n'
    s+='\\begin{tabular}{@{}l'+'r'*(len(cols)-1)+'@{}}\\toprule\n'+' & '.join(cols)+r'\\'+'\n\\midrule\n'
    s+='\n'.join(' & '.join(row)+r'\\' for row in rows)
    s+='\n\\bottomrule\\end{tabular}\n\\end{table}'
    save(name,s)
methods=['onehop','raw_max','orth_degree_max']
names={'onehop':'One-hop','raw_max':'Raw bank','orth_degree_max':'Controlled bank'}
mechanisms=['hard2','hard3','distance_decay','offgrid_0.1','offgrid_0.95','threshold','independent_cascade']
short=['2-hop','3-hop','D',r'$\delta=.1$',r'$\delta=.95$','T','IC']
pool=source('round3_confirmation_pooled')
null=source('round2_null_pooled')
rows=[]
for method in methods:
    parts=[]
    for design in ['Bernoulli','complete']:
        q=null.query('design==@design and method==@method').iloc[0]
        parts.extend([f(100*q.rate),ci(100*q.ci_low,100*q.ci_high)])
    rows.append([names[method],*parts])
tab('calibration.tex','tab:null',
    ['Method','Bernoulli',r'95\% CI','Fixed count',r'95\% CI'],rows)
rows=[]
for design in ['Bernoulli','complete']:
    rows.append([r'\multicolumn{8}{l}{\textit{'+('Bernoulli' if design=='Bernoulli' else 'Fixed count')+r'}}'])
    for method in methods:
        q=pool.query('design==@design and method==@method').set_index('mechanism')
        rows.append([names[method],*[f(q.loc[m,'rate']) for m in mechanisms]])
tab('mechanisms.tex','tab:mechanisms',['Method',*short],rows)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(6.6,2.35),gridspec_kw={'width_ratios':[1,1,1.25]})
for ax,pos,edges in [(axes[0],np.array([[.12,.5],[.48,.5],[.84,.8],[.84,.2]]),[(0,1),(1,2),(1,3)]),
                     (axes[1],np.array([[.15,.5],[.75,.8],[.75,.2]]),[(0,1),(1,2),(0,2)])]:
    for i,j in edges:ax.plot(pos[[i,j],0],pos[[i,j],1],color='.6',lw=1.5,zorder=0)
    for i,j in [(0,1),(1,2)]:
        start,end=pos[i],pos[j]
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color='#D55E00',lw=2,shrinkA=9,shrinkB=9))
    ax.scatter(pos[:,0],pos[:,1],s=180,c=['#0072B2']+['#009E73']*(len(pos)-1),zorder=3)
    for xy,label in zip(pos,['F']+['C']*(len(pos)-1)):ax.text(*xy,label,ha='center',va='center',color='white',fontsize=8,zorder=4)
    ax.set_xlim(0,1);ax.set_ylim(-.3,1);ax.axis('off')
axes[0].set_title('(a) Locally tree-like',fontsize=9)
axes[0].text(.5,-.12,'Two steps reach new vertices\nTree-limit cosine: '+r'$1/\sqrt{d}$',ha='center',va='center',fontsize=8)
axes[1].set_title('(b) Clustered',fontsize=9)
axes[1].text(.5,-.12,'Two steps reach an existing\ndirect neighbor',ha='center',va='center',fontsize=8)
from scipy.stats import binom
counts=np.arange(25,73)
axes[2].bar(counts,binom.pmf(counts,120,.4),color='#0072B2',alpha=.75,label='Bernoulli')
axes[2].axvline(48,color='#D55E00',lw=2,label='Fixed count')
axes[2].set_title('(c) Assignment design',fontsize=9)
axes[2].set_xlabel('Number treated in C',fontsize=8);axes[2].set_ylabel('Bernoulli probability',fontsize=8)
axes[2].tick_params(labelsize=7);axes[2].legend(fontsize=7,loc='upper right')
fig.tight_layout(pad=.6);fig.savefig(ROOT/'figures/schematic.png',dpi=240);plt.close(fig)
diagnostics=pd.read_csv(ROOT/'rewiring_diagnostics.csv') if (ROOT/'rewiring_diagnostics.csv').exists() else source('rewiring_moment_diagnostics')
rows=[]
for method in ['onehop','matched_kernel']:
    for p,q in diagnostics.query('method==@method').groupby('p'):
        finite=q.moment_theta80.notna()
        rows.append([('1' if method=='onehop' else 'M'),f(p),f(q.cosine.mean()),f(q.gamma_infinity.mean()),f(q.variance_ratio_at_theta1.mean()),str(int(finite.sum()))+'/6',
                     f(q.observed_theta80.mean()),f(q.weak_theta80.mean()),
                     f(q.moment_theta80.mean()) if finite.any() else '--',
                     f(100*q.relative_error.abs().mean()) if finite.any() else '--'])
tab('threshold_predictions.tex','tab:predictions',
    ['Score','Rewiring','Cosine',r'$\Gamma_\infty$',r'$r$','Finite','Observed','Weak','Pred.','MAPE (\\%)'],rows)
def village_source(name):
    p=ROOT/'village_output'/f'{name}.csv'
    return pd.read_csv(p) if p.exists() else source(name)
vgeom=village_source('village_geometry');veff=village_source('village_efficiency')
tab('village_geometry.tex','tab:vgeometry',
    ['Village',r'$n_0$',r'$n$','Degree','Clustering'],
    [[str(int(r.village)),str(int(r.n_original)),str(int(r.n)),f(r.mean_degree),f(r.clustering)] for r in vgeom[vgeom.village.isin(veff.village.unique())].itertuples()])
rows=[]
for v in veff.village.drop_duplicates():
    q=veff[veff.village==v].set_index(['scenario','method'])
    one,matched=q.loc[('onehop','onehop')],q.loc[('onehop','matched_kernel')]
    # Under the one-hop response the two tests are the same statistic up to scale.
    assert np.isclose(one.predicted_theta80,matched.predicted_theta80)
    assert np.isclose(one.observed_theta80,matched.observed_theta80)
    obs=lambda r:f(r.observed_theta80) if pd.notna(r.observed_theta80) else r'$>40$'
    row=[str(int(v)),f(one.predicted_theta80),obs(one)]
    for scenario in ['twohop','decay']:
        for method in ['onehop','matched_kernel']:
            r=q.loc[(scenario,method)]
            assert pd.isna(r.predicted_theta80)
            row.append(obs(r))
    rows.append(row)
s='\\begin{table}[htbp]\n\\centering\\small\\setlength{\\tabcolsep}{4pt}\n'
s+='\\caption{}\\label{tab:vthresholds}\n'
s+='\\begin{tabular}{@{}lrrrrrr@{}}\\toprule\n'
s+=r'& \multicolumn{2}{c}{One-hop response} & \multicolumn{2}{c}{Two-hop response} & \multicolumn{2}{c}{Decay response}\\'+'\n'
s+=r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}'+'\n'
s+=r'Village & Pred. & Obs. & Obs. 1 & Obs. M & Obs. 1 & Obs. M\\'+'\n\\midrule\n'
s+='\n'.join(' & '.join(row)+r'\\' for row in rows)
s+='\n\\bottomrule\\end{tabular}\n\\end{table}'
save('village_thresholds.tex',s)
fig,(left,right)=plt.subplots(1,2,figsize=(6.6,2.8),gridspec_kw={'width_ratios':[1,1.55]})
local=veff.query('method=="onehop" and scenario=="onehop"')
left.plot([.5,1.25],[.5,1.25],'--',color='.6',lw=1)
left.errorbar(local.predicted_theta80,local.observed_theta80,
    yerr=np.array([local.observed_theta80-local.theta80_ci_low,local.theta80_ci_high-local.observed_theta80]),
    fmt='o',color='#0072B2',capsize=2,markersize=4)
offsets={1:(-19,10),4:(5,-10),8:(4,-8),12:(7,6),17:(10,-12),21:(-16,-9)}
for r in local.itertuples():left.annotate(str(int(r.village)),(r.predicted_theta80,r.observed_theta80),xytext=offsets[int(r.village)],textcoords='offset points',fontsize=7)
left.set_xlabel(r'Predicted $\theta_{80}$');left.set_ylabel(r'Observed $\theta_{80}$')
left.set_title('(a) One-hop response',fontsize=9)
villages=sorted(veff.village.unique())
for method,color,offset in [('onehop','#444444',-.12),('matched_kernel','#009E73',.12)]:
    for scenario,marker in [('twohop','o'),('decay','^')]:
        q=veff.query('method==@method and scenario==@scenario').set_index('village').loc[villages]
        right.scatter(np.arange(len(villages))+offset,q.max_power,marker=marker,color=color,s=24,
            label=('1' if method=='onehop' else 'M')+': '+('2-hop' if scenario=='twohop' else 'decay'))
right.axhline(.8,color='.5',ls='--',lw=1)
right.set_xticks(np.arange(len(villages)),[str(int(v)) for v in villages]);right.set_ylim(0,1)
right.set_xlabel('Village');right.set_ylabel('Maximum grid power')
right.set_title('(b) Broader responses: all predictions absent',fontsize=8)
for ax in [left,right]:ax.grid(axis='y',alpha=.2)
fig.tight_layout(pad=.6);fig.savefig(ROOT/'figures/villages.png',dpi=240);plt.close(fig)
colors={'onehop':'#444444','raw_max':'#0072B2','orth_degree_max':'#D55E00','matched_kernel':'#009E73'}
ws=source('round3_ws_intervals')
fig=plt.figure(figsize=(6.6,4.7))
gs=fig.add_gridspec(2,2,width_ratios=[1,7],height_ratios=[3,2],wspace=.09,hspace=.34)
ax0=fig.add_subplot(gs[0,0]);ax=fig.add_subplot(gs[0,1],sharey=ax0)
bottom0=fig.add_subplot(gs[1,0]);bottom=fig.add_subplot(gs[1,1],sharey=bottom0)
for method in ['onehop','raw_max','orth_degree_max','matched_kernel']:
    q=ws.query('method==@method and quantity=="theta80"').sort_values('p')
    assert (q.observed==6).all() and (q.total==6).all()
    for a,z in [(ax0,q[q.p==0]),(ax,q[q.p>0])]:
        a.errorbar(z.p if a==ax else np.zeros(len(z)),z['mean'],
          yerr=np.array([z['mean']-z.ci_low,z.ci_high-z['mean']]),fmt='o-',capsize=2,
          markersize=4,color=colors[method],label=names.get(method,'Matched statistic'))
ax.set_xscale('log');ax.set_xticks([.01,.02,.05,.1,.2,.5,1],[r'.01','.02','.05','.1','.2','.5','1'])
ax.tick_params(labelleft=False,left=False);ax.spines['left'].set_visible(False)
ax0.set_xticks([0],['0']);ax0.set_xlim(-.4,.4);ax0.set_ylabel(r'Signal strength $\theta_{80}$')
for method in ['onehop','matched_kernel']:
    q=diagnostics.query('method==@method').groupby('p').agg(pred=('moment_theta80','mean'),count=('moment_theta80','count'))
    q=q[q['count']==6]
    for a,z in [(ax0,q[q.index==0]),(ax,q[q.index>0])]:
        a.plot(z.index if a==ax else np.zeros(len(z)),z.pred,'--',lw=1.3,color=colors[method],
               label=('One-hop moment' if method=='onehop' else 'Matched moment'))
ax.legend(fontsize=6.9,loc='upper left',ncol=2);ax.set_title('(a) Observed and moment-predicted thresholds',fontsize=9)
target=diagnostics.normal_shift_for_80.iloc[0]
for method in ['onehop','matched_kernel']:
    q=diagnostics.query('method==@method')
    for a,z in [(bottom0,q[q.p==0]),(bottom,q[q.p>0])]:
        a.scatter(z.p if a==bottom else np.zeros(len(z)),z.gamma_infinity,s=9,alpha=.45,color=colors[method])
        mean=z.groupby('p').gamma_infinity.mean()
        a.plot(mean.index if a==bottom else np.zeros(len(mean)),mean.values,'o-',ms=3,color=colors[method])
for a in [bottom0,bottom]:a.axhline(target,color='.4',ls='--',lw=1)
bottom.set_title('(b) Graphwise moment limits; dashed level 2.80',fontsize=9)
bottom0.set_ylabel(r'$\Gamma_{K,\infty}$');bottom.set_xlabel('WS rewiring probability')
for a in [ax,bottom]:
    a.set_xscale('log');a.set_xticks([.01,.02,.05,.1,.2,.5,1],['.01','.02','.05','.1','.2','.5','1'])
    a.tick_params(labelleft=False,left=False);a.spines['left'].set_visible(False)
for a in [ax0,bottom0]:a.set_xticks([0],['0']);a.set_xlim(-.4,.4)
for a in [ax0,ax,bottom0,bottom]:a.grid(axis='y',alpha=.2)
fig.subplots_adjust(left=.12,right=.98,bottom=.11,top=.96)
fig.savefig(ROOT/'figures/rewiring.png',dpi=240);plt.close(fig)

labels={'Regular':'Regular','ER':'ER','BA':'BA','SBM_003':'SBM(.03)','SBM_045':'SBM(.45)','WS_002':'WS(.02)','WS_020':'WS(.2)','WS_100':'WS(1)'}
summary=source('round3_confirmation_summary')
path=ROOT/'neighbourhood_expansion.csv'
expansion=pd.read_csv(path) if path.exists() else source('neighbourhood_expansion')
expansion['family']=expansion.kind.map(lambda k:'WS' if k.startswith('WS') else 'SBM' if k.startswith('SBM') else 'Villages' if k=='Village' else k)
fig,ax=plt.subplots(figsize=(5.5,2.8))
x=np.linspace(0,expansion.global_expansion_ratio.max()+.5,300)
palette={'Regular':'#222222','ER':'#0072B2','BA':'#E69F00','SBM':'#CC79A7','WS':'#009E73','Villages':'#D55E00'}
ax.plot(x,1/np.sqrt(1+x),'--',color='.45',lw=1,label='Count-only benchmark')
for family,color in palette.items():
    q=expansion[expansion.family==family]
    ax.scatter(q.global_expansion_ratio,q.cosine,s=15,alpha=.7,color=color,label=family,
        marker='^' if family=='Villages' else 'o')
ax.set_xlabel(r'Distinct second-neighbour expansion $z_2/z_1$')
ax.set_ylabel('One-hop overlap');ax.grid(alpha=.16)
ax.legend(fontsize=6.7,ncol=2,loc='upper right');ax.set_ylim(.2,1.02)
fig.tight_layout(pad=.6);fig.savefig(ROOT/'figures/expansion.png',dpi=240);plt.close(fig)

local=source('round4_local_summary')
fig,axes=plt.subplots(2,2,figsize=(6.6,3.8),sharex=True,sharey=True)
for ax,(n,d) in zip(axes.flat,[(1000,4),(1000,8),(5000,4),(5000,8)]):
    q=local.query('n==@n and d==@d')
    theory=q.drop_duplicates('t').sort_values('t')
    ax.plot(theory.t,theory.theory,'k-',label='Local limit')
    for method,shift,marker,color,label in [('raw_two',-.08,'o','#0072B2','Raw two'),('controlled_two',.08,'^','#D55E00','Controlled two')]:
        x=q.query('method==@method').sort_values('t')
        ax.errorbar(x.t+shift,x.rate,yerr=np.array([x.rate-x.ci_low,x.ci_high-x.rate]),fmt=marker,
            color=color,capsize=2,markersize=4,label=label)
    ax.set_title(f'n={n}, d={d}',fontsize=10);ax.grid(alpha=.18);ax.set_ylim(0,1)
axes[0,0].legend(fontsize=7,loc='upper left')
for ax in axes[1,:]: ax.set_xlabel('Local strength t')
for ax in axes[:,0]: ax.set_ylabel('Rejection probability')
fig.tight_layout();fig.savefig(ROOT/'figures/local_power.png',dpi=240);plt.close(fig)

rows=[]
for design in ['Bernoulli','complete']:
    method='orth_degree_max' if design=='Bernoulli' else 'raw_max'
    for m,label in zip(mechanisms,short):
        q=pool.query('design==@design and method==@method and mechanism==@m').iloc[0]
        rows.append(['B' if design=='Bernoulli' else 'F',label,f(q.rate),ci(q.ci_low,q.ci_high),f(100*q.gain),ci(100*q.gain_ci_low,100*q.gain_ci_high)])
tab('paired.tex','tab:spaired',['Design','Response','Power',r'95\% CI','Gain (pp)',r'95\% CI (pp)'],rows)
s=r'\small\setlength{\tabcolsep}{2pt}'+'\n'+r'\begin{longtable}{@{}llrrrlrrrl@{}}'+'\n'
s+=r'\caption{}\label{tab:sconfirm}\\'+'\n'
header=(r'\toprule & & \multicolumn{4}{c}{Bernoulli: designated C} & \multicolumn{4}{c}{Fixed count: designated R}\\'
        r'\cmidrule(lr){3-6}\cmidrule(lr){7-10}Graph & Response & 1 & R & C & Gain [95\% CI] & 1 & R & C & Gain [95\% CI]\\\midrule')
s+=header+'\n'+r'\endfirsthead'+'\n'+header+'\n'+r'\endhead'+'\n'
for kind in ['ER','Regular','BA','SBM_003','SBM_045','WS_002','WS_020','WS_100']:
    for i,(m,ml) in enumerate(zip(mechanisms,short)):
        row=[labels[kind] if i==0 else '',ml]
        for design in ['Bernoulli','complete']:
            q=summary.query('design==@design and kind==@kind and mechanism==@m').set_index('method')
            main='orth_degree_max' if design=='Bernoulli' else 'raw_max'
            row.extend([*[f(q.loc[method,'rate']) for method in methods],f(100*q.loc[main,'gain'])+' '+ci(100*q.loc[main,'gain_ci_low'],100*q.loc[main,'gain_ci_high'])])
        s+=' & '.join(row)+r'\\'+'\n'
    s+=r'\addlinespace[2pt]'+'\n'
s+=r'\bottomrule\end{longtable}\normalsize'+'\n'
save('confirmation_full.tex',s)
print('Generated five figures and seven table source files in figures/ and tables/.')
