"""Dense, outcome-free moment diagnostics for homogeneous Bernoulli assignment.

Assumptions: fixed undirected unweighted graph, disjoint outcome-free roles,
Y = theta H(X_C-p) + iid mean-zero noise of known standard deviation.
The Gaussian shift approximation is for one two-sided score, not a max bank.
An absent moment crossing is NOT an infinite threshold of the actual test.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.sparse.csgraph import shortest_path

def response_operator(A,F,C,scenario='twohop'):
    """Candidate FC response, normalized to squared Frobenius norm |F|."""
    a=np.asarray(A,dtype=float)
    degree=a.sum(axis=1)
    inv=np.divide(1,np.sqrt(degree),out=np.zeros_like(degree),where=degree>0)
    if scenario=='onehop':h=(inv[:,None]*a*inv[None,:])[np.ix_(F,C)]
    else:
        distance=shortest_path(a,directed=False)[np.ix_(F,C)]
        if scenario=='twohop':h=(distance<=2).astype(float)
        elif scenario=='decay':h=np.where(np.isfinite(distance),np.exp(-(distance-1)/1.5),0.)
        else:raise ValueError('scenario must be onehop, twohop or decay')
    energy=np.sum(h*h)
    return h*np.sqrt(len(F)/energy) if energy>0 else h

def predict_threshold(A,F,C,H,p=.4,noise_sd=1.,alpha=.05,target_power=.8,kernel='onehop'):
    """Return diagnostics and an optional finite single-score amplitude prediction.

    H is a specified |F| x |C| response, including its chosen amplitude scale.
    noise_sd must be positive; alpha < target_power < 1. No fitted outcomes
    enter the calculation. Suitable for small/medium dense networks only.
    kernel='matched' uses H; kernel='onehop' uses symmetric normalization.
    """
    a=np.asarray(A,dtype=float);f=np.asarray(F,dtype=int);c=np.asarray(C,dtype=int)
    h=np.asarray(H,dtype=float)
    if a.ndim!=2 or a.shape[0]!=a.shape[1]:raise ValueError('A must be square')
    if not np.isfinite(a).all() or not np.allclose(a,a.T):raise ValueError('A must be finite and symmetric')
    if not np.isin(a,[0,1]).all() or np.any(np.diag(a)):raise ValueError('Use an unweighted graph with no self-loops')
    if len(np.unique(f))!=len(f) or len(np.unique(c))!=len(c) or np.intersect1d(f,c).size:raise ValueError('F and C must be disjoint and contain no duplicates')
    if min(len(f),len(c))==0 or min(f.min(),c.min())<0 or max(f.max(),c.max())>=len(a):raise ValueError('Invalid role indices')
    if h.shape!=(len(f),len(c)) or not np.isfinite(h).all():raise ValueError('Invalid response block')
    if not (0<p<1 and noise_sd>0 and 0<alpha<target_power<1):raise ValueError('Invalid design, noise or power parameters')
    degree=a.sum(axis=1);inv=np.divide(1,np.sqrt(degree),out=np.zeros_like(degree),where=degree>0)
    if kernel=='onehop':k=(inv[:,None]*a*inv[None,:])[np.ix_(f,c)]
    elif kernel=='matched':k=h
    else:raise ValueError('kernel must be onehop or matched')
    critical=norm.ppf(1-alpha/2)
    shift=brentq(lambda g:norm.sf(critical-g)+norm.cdf(-critical-g)-target_power,0,critical+12)
    kn=np.linalg.norm(k);hn=np.linalg.norm(h);product=np.linalg.norm(k.T@h)
    if min(kn,hn,product)<=1e-14:
        return dict(status='degenerate',theta=None,cosine=0.,gamma_infinity=0.,variance_ratio_at_theta1=0.,weak_theta=None,normal_shift=shift)
    inner=abs(np.sum(k*h));gamma=inner/product
    ratio=p*(1-p)*product**2/(noise_sd**2*kn**2)
    theta=shift/np.sqrt(ratio*(gamma**2-shift**2)) if gamma>shift else None
    weak=shift/(gamma*np.sqrt(ratio)) if gamma>0 else None
    return dict(status='finite' if theta is not None else 'no_moment_crossing',theta=theta,
                cosine=inner/(kn*hn),gamma_infinity=gamma,variance_ratio_at_theta1=ratio,
                weak_theta=weak,normal_shift=shift)
