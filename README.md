# Spectral geometry and power of randomization tests for network interference

Reproducibility code and aggregate results for the manuscript.

Repository: https://github.com/ouruocun-dotcom/network-interference-tests

## Contents

- code/: simulation, statistical verification and figure-generation scripts.
- requirements.txt: Python dependencies.
- aggregate_results.csv: aggregate and graph-level reference results.

No participant records, third-party raw datasets, permutation arrays or
individual adoption trajectories are included.

## Installation

Use Python 3.11 or later:

~~~bash
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
~~~

Run commands from the repository root. Full simulations can take substantial
time. Where available, --quick checks program operation rather than reproducing
the reported experiments.

## Primary analyses

Select the primary bank by assignment design before inspecting outcomes:
fixed-count or block-complete assignment uses raw_max; independent Bernoulli
assignment uses orth_degree_max. The latter retains adjacency and residualizes
tails against adjacency and the global mode.

~~~bash
python code/revision_round3.py --output results/confirmation
python code/revision_efficiency3.py --mode ws --output results/ws
python code/revision_efficiency3.py --mode size --output results/size
python code/revision_limits.py --output results/local_power
~~~

Independent confirmation uses seeds 2026100931/32, twelve graphs per ensemble
and 64 datasets per graph. Rewiring and size studies use seeds 2026100933/34.
Local-power verification uses seed 2026100941, n=1000/5000, d=4/8, six graphs
per cell, 512 datasets per graph and 499 reassignments.

The strength threshold is the first interpolated crossing of the
cumulative-maximum power envelope at 80%. Missing thresholds are censored;
this does not guarantee sustained power at larger amplitudes. The matched
kernel is a comparator, not a power upper bound.

## Mechanism comparisons and calibration

~~~bash
python code/revision_study.py --output results/development
python code/revision_round2.py --mode null --output results/calibration
python code/revision_round2.py --mode bernoulli --output results/design
python code/revision_round2.py --mode complete --output results/design
python code/revision_round2.py --mode regret --output results/design
python code/revision_size.py --output results/fixed_strength
python code/revision_controlled.py --output results/rewiring
python code/revision_nonlinear.py --output results/nonlinear
python code/revision_tree.py
python code/revision_study.py --scaling --output results/timing
~~~

The null calibration study has 20,480 datasets per assignment design.
Comparisons share outcomes and reassignments within a dataset; paired
intervals cluster by graph. Positive-direction benchmarks are separate
from primary two-sided tests. Complete-design split-Fisher statistics are
jointly rank-calibrated because counts in their halves are dependent.

## Spectral and normalization studies

~~~bash
python code/effective_rank_audit.py --output results/spectral_checks.csv
python code/unpack_spectral_results.py aggregate_results.csv results/spectral_reference
python code/detectability/null_full.py code results/normalized_null
python code/detectability/bank8.py code results/normalized_banks
python code/detectability/size240_rerun.py code results/size240 1024
python code/detectability/effective_rank_size.py code results/spectra240 0 240
python code/detectability/make_figure.py results/spectral_reference results
~~~

The verification driver uses seed 2026101047 and includes exhaustive
fixed-count variance checks, equal-spectrum simulations, counterexamples
and scoring checks. Effective-rank divergence is sufficient, not necessary.
The fixed-rank Gaussian formula requires diffuse singular vectors.
Assignment normalization is an extension, not the primary bank rule.

scores_fast.py implements the convention used in the network reference
results, including an absolute denominator threshold of 1e-12. scores_safe.py
provides scale-stable zero handling. These are distinct arithmetic conventions;
reference results must not be attributed to a different implementation.

In the spectral-size driver, MC=0 selects counts 2000/1000/300 by network
size; positive MC sets the count explicitly. limit_matched_mc denotes a
noise-free finite-randomization calculation. chi2_pred on full-rank blocks
is a Gaussian working approximation, not an unconditional application of
the fixed-rank theorem.

## Result format

source_table identifies the study and result table. Inapplicable fields are
empty. Script and output names are retained for command and schema compatibility.
spectral_fields and spectral_relative_path record the columns and relative
paths of spectral outputs. The unpacking command restores separate tables
without numerical conversion. spectral_independent and spectral_supplied__*
distinguish verification checks from network-study tables. Local runs write
to the selected directory instead of overwriting the reference CSV.

## Optional school illustration

The supplementary school illustration requires ICPSR study 37070 version 2
(DOI: 10.3886/ICPSR37070.v2). Obtain it independently under the repository's
terms. Do not upload participant records or decoded caches.

~~~bash
python code/revision_field.py --data /absolute/path/37070-0001-Data.rda
~~~

This exploratory calculation is conditional on the reconstructed assignment
design, not confirmatory method-validation evidence. The 815 records with
unknown treatment/stratum fields are not imputed untreated. Endpoint
selection and roster assumptions are described in the Supplementary Information.
