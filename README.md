# Spectral geometry and power of network randomization tests

## Manuscript structure and proof clarification

The current editorial revision separates numerical study design from results,
concentrates limitations in the discussion, and adds a conclusion. The abstract
has 210 whitespace-delimited words. SI S15 explicitly bounds the conditional
reference tails and observed-statistic sandwich, handles the shrinking
neighbourhood in the counterexample, and distinguishes asymptotic coefficient
signs from finite-sample signs. No numerical constant for Berry--Esseen is
required. Numerical outputs and analysis code are unchanged from v9.
The author reports an additional independent S15 proof audit; its scripts were
not supplied here, and it is not a rerun of S16 or a final approval declaration.

One ZIP contains ordinary readable Python sources, pinned requirements and
one aggregate-results CSV. The `source_table` column identifies each table;
inapplicable fields are empty. No participant records or external raw inputs,
permutation arrays or individual trajectories are redistributed.

## Primary analysis and independent confirmation

Choose before inspecting outcomes: fixed-count or block-complete assignment
uses `raw_max`; independent Bernoulli assignment uses `orth_degree_max`
(tails residualized against W and the global mode G, while retaining W).
Neither is a test of the null permitting arbitrary one-hop interference.
The rule was developed after earlier grids, then fixed before independent
confirmation using seeds 2026100931/32. There is no preregistration claim.

Run these commands from the unpacked root:

```bash
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python code/revision_round3.py --output results/confirmation
python code/revision_efficiency3.py --mode ws --output results/ws
python code/revision_efficiency3.py --mode size --output results/size
```

Confirmation has twelve graphs per ensemble and 64 outcomes per graph under
each design. WS uses six graphs and 256 outcomes per rewiring value, fourteen
strengths through 16. Size curves use Bernoulli(.4), eight graphs and 64
outcomes per cell at n=240/1000/5000. Both strength studies use interpolated
first crossings of interpolated cumulative-maximum power envelopes at 80%; censored values remain missing and graph
denominators are recorded. An uncensored-subset mean is not an ensemble mean.
These thresholds do not promise sustained power at all larger amplitudes;
the original small-sample ER Hard-3 raw curve declines at high strength.
A supplied 1,024-outcome-per-graph rerun does not reproduce its largest-strength
estimate below nominal. Original and supplied-rerun rows remain separate. Studentization
can produce structural saturation. The limiting conditional critical value
is assignment-dependent, not universally a fixed Gaussian quantile.
K=H is a matched kernel, not a power upper bound. Some stored older tables
use the method key `oracle`; this does not change that interpretation.
Sparse coefficients use 32 graph-only probes, not exact Frobenius projection.

The WS wrapper reuses the tested curve engine, so intermediate files have
`round2_controlled_*` names. The `round3_ws_*` outputs are the authoritative
new WS tables, with matched-kernel names and explicit censoring fields.
Reference CSVs are not overwritten by these commands. `--quick` only tests
program operation, not the scientific conclusions.

## Two-component local-power verification

```bash
python code/revision_limits.py --output results/local_power
```

Seed 2026100941; n=1000/5000, d=4/8, six regular graphs per cell,
512 datasets per graph, B=499; t=0,2,4,6,8 and theta=t/sqrt(|F|).
Common draws/noise compare precisely {W_FC,(W^2)_FC} and its W/G-controlled
counterpart, not the full seven-kernel banks. Formula discrepancies and
exact graph-only projection shares are supplied. The formula assumes
Gaussian noise, fixed degree and n,B tending to infinity. No finite-B
exact-power claim is made. A noise-free weak-rank calculation need not
be the limit of finite-strength ranks on discrete ties or degenerate weights.

## Development studies and calibration

```bash
python code/revision_study.py --output results/development
python code/revision_round2.py --mode null --output results/audit
python code/revision_round2.py --mode bernoulli --output results/design
python code/revision_round2.py --mode complete --output results/design
python code/revision_round2.py --mode regret --output results/design
python code/revision_size.py --output results/fixed_strength
python code/revision_controlled.py --output results/rewiring_development
python code/revision_nonlinear.py --output results/nonlinear
python code/revision_tree.py
python code/revision_study.py --scaling --output results/timing
```

The large audit has 20,480 null datasets per design. Development comparisons
use outcome-independent random auxiliary halves. Complete-design split Fisher
is jointly rank calibrated because counts in its halves are dependent.
The original initial-grid driver retains a vertex-order split to reproduce
that stored comparison; it is not the primary CRT implementation. Zero-variance
components contribute zero. Positive-direction benchmarks are separate from
the primary two-sided tests. The local-tree moment limit is distinct from
the two-component local-power formula, which also assumes Gaussian noise,
local signals, fixed degree and B tending to infinity.

## Spectral extension and its limits

Effective-rank divergence is sufficient, not necessary. The fixed-rank Gaussian
law requires fixed singular values and diffuse right singular vectors. Effective
rank does not determine power or bound all statistics. The fixed-count quadratic
variance identity gives a sufficient bound; the fixed-count versions of the
Bernoulli Gaussian law and anti-concentration rate are not proved. Assignment
normalization is an extension, not the independently confirmed primary bank rule.
Rank at least two alone does not ensure consistency on finite lattice designs.

```bash
python code/effective_rank_audit.py --output results/independent_checks.csv
python code/unpack_spectral_results.py aggregate_results.csv results/spectral_reference
python code/detectability/null_full.py code results/normalized_null
python code/detectability/bank8.py code results/normalized_banks
python code/detectability/size240_rerun.py code results/size240 1024
python code/detectability/effective_rank_size.py code results/spectra240 0 240
python code/detectability/make_figure.py results/spectral_reference results
```

The independent driver uses seed 2026101047: six exhaustive variance checks,
equal-spectrum simulations (4,096 datasets per rank, B=499), a bounded-effective-
rank consistency counterexample, a rank-two lattice counterexample, and scoring,
zero-kernel and scale checks. It is not a full rerun of supplied network studies.
`spectral_independent` rows are new checks; `spectral_supplied__*` rows are supplied
reference outputs. `spectral_fields` and `spectral_relative_path` preserve their
schemas for unpacking without numeric conversion. The 101,954 original retained
rows are unchanged; additional columns on them are empty.

MC=0 in the spectral-size driver selects original by-size counts 2000/1000/300;
MC>0 now actually overrides them. The size240 rerun handles partial batches.
These fixes do not alter original default calculations. `limit_matched_mc`
denotes noise-free finite-B=499 power, not an ideal population limit.
`chi2_pred` on full-rank network blocks is a Gaussian approximation; optional
equal-spectrum curves are references, not universal bounds.

Supplied studies used the absolute denominator cutoff 1e-12 in `scores_fast.py`;
that legacy function is retained for reproducibility. `scores_safe.py` corrects
scale-dependent zero handling. Independent checks verify ordinary scales,
rescaling by 1e-15 and zero kernels. Full network simulations have not been
rerun with corrected arithmetic and must not be relabelled as such.
Both fixed-count one-hop null intervals exceed 0.05. The supplied follow-up
checks only studentized scores on three ensembles, not every implementation.
No new held-out confirmation of a normalized-bank recommendation is claimed.

The optional SymPy-only symbolic script and obsolete LaTeX template filler
are not shipped: the SI has the full coincidence-pattern derivation and the
independent NumPy/SciPy driver checks it. No additional dependency is required.
Original/supplied/new results remain distinguished; reference snapshots are
not overwritten by simulation commands.

## Optional conditional school demonstration (SI only)

ICPSR study 37070 version 2 is public-use and requires no member affiliation.
Obtain it independently and accept its current repository terms.
Use for research/statistics only, protect confidentiality, cite
10.3886/ICPSR37070.v2 and report the publication citation to ICPSR.
The source records and repository materials must not be publicly redistributed
without the required permission.

Obtain the data under its repository terms:

```bash
python code/revision_field.py --data /absolute/path/37070-0001-Data.rda
```

All sensitivities reuse 49,999 block-preserving assignments, including
undirected row normalization. The endpoint and adjacency-residual bank are
not reselected. The 815 unknown-treatment records are not imputed untreated. They all lack
TREAT/STRB; 756 lack first-wave survey type, while 807 have normal second-wave
survey type. Codebook 'not applicable' gives no cause. TREAT=0 means neither
treatment nor control, not randomized control. Matching the published 728
seed total is an aggregate check, not proof of complete eligible blocks.
The illustration is confined to the SI and is not method-validation evidence.
Outcome exploration and unverified roster/design assumptions limit causal
interpretation; the calculation is conditional on the reconstructed design. Conditional-tail Monte Carlo intervals do not adjust for
endpoint selection.
