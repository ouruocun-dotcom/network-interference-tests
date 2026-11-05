# Second-neighbour expansion and the detection of network spillovers

Code and aggregate results for the manuscript.
https://github.com/ouruocun-dotcom/network-interference-tests

## Install and reproduce the figures and tables

Use Python 3.12 (Python 3.11 is also supported). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python code/build_assets.py
```

On Windows, activate with `.venv\Scripts\activate` and set the two thread
variables using the syntax of your shell. The pinned requirements specify the
numerical and plotting libraries. No LaTeX installation is needed to generate
figures and table source files.

The last command reads `aggregate_results.csv` and produces the three main
figures, two supplementary figures and seven table source files (Table 1 and
Tables S1–S6). It runs without downloading village data or rerunning simulations.
The table captions match the accompanying manuscript. The CSV retains full
precision; displayed values are rounded to two decimal places.

## Outputs and calculation drivers

| Manuscript output | Driver | CSV source_table |
| --- | --- | --- |
| Figure 1, network/design schematic | `build_assets.py` | Illustration |
| Figure 2, second-neighbour expansion | `neighbourhood_expansion.py` | `neighbourhood_expansion` |
| Figure 3, WS thresholds | `revision_efficiency3.py --mode ws` | `round3_ws_intervals` |
| Figure S1, local power | `revision_limits.py` | `round4_local_summary` |
| Figure S2, village thresholds | `village_study.py` | `village_efficiency` |
| Table 1 and Tables S1–S2, confirmation | `revision_round3.py` | `round3_confirmation_pooled`, `round3_confirmation_summary` |
| Table S3, moment diagnostics/predictions | `diagnose_rewiring.py` | `rewiring_moment_diagnostics` |
| Tables S4–S5, village profiles/thresholds | `village_study.py` | `village_geometry`, `village_efficiency` |
| Table S6, null calibration | `revision_round2.py --mode null` | `round2_null_pooled` |
| Nonlinear comparison | `revision_nonlinear.py` | `round2_nonlinear_summary` |
| Degree sensitivity | `degree_diagnostics.py` | `ws_degree_diagnostics` |
| Tree alignment | `revision_tree.py` | `round2_tree_summary` |
| Size comparison | `revision_efficiency3.py --mode size` | `round3_size_*` |
| Additional method comparisons | `revision_round2.py --mode bernoulli` / `--mode complete` | `round2_bernoulli_pooled`, `round2_complete_pooled` |
| Sparse computation timing | `revision_study.py --scaling` | `revision_scaling` |

Driver names in this table are relative to `code/`. The CSV combines several
schemas: `source_table` identifies the calculation, and inapplicable fields are
empty. Graph-level rates, pooled estimates, paired differences and intervals
are stored separately. Extra method columns are retained to preserve complete
comparisons. `raw_max` denotes the raw bank and `orth_degree_max` the controlled
bank; `onehop` is the one-hop test and `matched_kernel` uses the response itself.

## Regenerate calculations

Full simulation runs can be expensive. These commands regenerate calculations
using the settings and seeds specified in the scripts:

```bash
python code/revision_round3.py --output results/confirmation
python code/revision_round2.py --mode null --output results/calibration
python code/revision_round2.py --mode bernoulli --output results/comparisons_B
python code/revision_round2.py --mode complete --output results/comparisons_F
python code/revision_efficiency3.py --mode ws --output results/rewiring
python code/revision_efficiency3.py --mode size --output results/size
python code/revision_limits.py --output results/local_power
python code/revision_nonlinear.py --output results/nonlinear
python code/revision_tree.py
python code/revision_study.py --scaling --output results/timing
python code/diagnose_rewiring.py
python code/degree_diagnostics.py
python code/village_study.py
python code/neighbourhood_expansion.py
```

For a short executable example of both assignment designs, run:

```bash
python code/revision_round3.py --quick --output results/example
```

`--quick` reduces sample sizes and does not reproduce reported power estimates.
Full reruns write new output files; they do not replace the supplied reference
CSV. `revision_study.py` supplies common kernels and mechanisms;
`revision_controlled.py` and `revision_size.py` also supply routines used by the
WS and size driver.

Confirmation uses 12 graphs per family, 64 datasets per graph and 499 fresh
re-randomizations; seeds are 2026100931/32. The null calculation uses 20,480
datasets per design and seed 2026100921. WS/size seeds are 2026100933/34;
local-power seed is 2026100941. Complete grids and further seeds are in each
calculation driver. Set the thread variables above for consistent timing.
Different library/platform arithmetic can affect extremely close numerical ties.

## Village network input

Source: Banerjee et al., *The Diffusion of Microfinance*, Harvard Dataverse,
https://doi.org/10.7910/DVN/U3BIHX, released version 9.4, CC0 1.0.
The download helper retrieves the authors' publicly linked Stanford mirror and
checks the archive size (16,037,457 bytes) and MD5
`a61cc1f89000677b46dcfbc492e81b64` before use. If automatic download is unavailable,
place that verified archive at `inputs/verified_network_data.zip`; alternatively
pass `--archive /path/to/datav4.0.zip` to `village_study.py`.

Only `adj_allVillageRelationships_HH_vilno_*.csv` members are read. The ALL
network is the union of surveyed relationships: a household edge exists when
any household member reports such a relationship with a member of the other
household. Matrices are made binary and symmetric, self-loops removed, and the
largest component retained; size ties use the smallest original row index.
Raw networks, linkage keys, attributes and real adoption outcomes are not
included in this repository. All analysed assignments and outcomes are simulated.

The first 20 available villages are 1–12 and 14–21. Six fixed scans use villages
1, 4, 8, 12, 17 and 21. Partition seed is `2026102501 + 104729*v`; assignment/noise
seed is `2026102502 + 65537*v`. Each scan uses 512 datasets, 499 re-randomizations,
unit Gaussian noise and amplitudes `0,.15,.3,.5,.75,1,1.5,2,3,4,6,8,12,16,24,40`.
Candidate response blocks are one-hop adjacency, distance at most two, and decay
`exp(-(distance-1)/1.5)`, each normalized to squared Frobenius norm `|F|`.
Assignments/noise are shared across methods, responses and amplitudes.
Bootstrap intervals use 1,000 dataset resamples with seed
`2026102503 + 8191*v + 101*s + j`, response index `s=0,1,2` and method `j=0,1`.

## Statistical conventions and diagnostic function

Tests use the two-sided calibrated maximum, level 0.05, with ties counted against
rejection. `theta80` is the first interpolated 0.8 crossing of the cumulative
maximum power envelope. Missing crossings remain censored. Bootstrap
noncrossings count as infinite. Graph-level comparisons use paired intervals;
nonlinear-grid intervals are pointwise. One-hop and matched scores coincide for
the one-hop response and are not independent replications.

With `code/` on the Python path, import `predict_threshold` and
`response_operator` from `network_diagnostics`. For example:

```python
from network_diagnostics import predict_threshold
result = predict_threshold(A, F, C, H, p=.4, noise_sd=1.,
                           alpha=.05, target_power=.8)
```

`H` must be a specified focal-by-auxiliary response with its intended scaling.
The diagnostic assumes homogeneous Bernoulli assignment, zero baseline and
independent mean-zero noise with the specified variance. It returns cosine,
moment limit, variance ratio, optional predicted strength and a status.
`kernel='matched'` selects the matched comparator. A missing moment prediction
is not proof that the actual test cannot reach the target. Fixed-count designs
and max-bank power are not covered by this single-score prediction.
