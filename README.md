# Saber

Code release for *A Closed-Loop Bio-Robotic Platform for Monitoring Liver Cancer Organoids
and Tumor-Microenvironment-Aware Drug Screening* (Cyborg and Bionic Systems).

## Overview

The platform treats continuous organoid monitoring as damage-budgeted information
acquisition. A dish of organoid cultures is observed by a robotic imaging and liquid-handling
head; every observation costs specimen integrity, and the question the release answers is
which observation to take next. The code implements the decision problem, not a product:

- the action set of Eq. (1) and the damage functional of Eq. (2)/(6), with the budget defined
  as the reference protocol's own damage over the same wall-clock window (Sec. 2.4), so every
  comparison in the results tables is iso-resource;
- the receding-horizon planner of Eq. (7) and Algorithm 2, with the Lagrangian multiplier
  updated from the running damage and the action set pruned inside the control window;
- the two-statement theory layer: the interior-optimum result of Theorem 1, the constant-factor
  greedy guarantee of Proposition 1 and Corollary 1, and the parity result of Theorem 2, each
  recomputed from its own closed form rather than asserted;
- the perception front end of Sec. 2.2 (two adapted backbones, instance masks, trajectories,
  morphology embeddings), the four-axis variational estimator of Sec. 2.3 and Algorithm 1, the
  microenvironment-conditioned response head of Sec. 2.6, the coupled growth/oxygen twin of
  Sec. 2.7, and the off-policy estimator family of Sec. 3.7.

The article's evaluation is a retrospective secondary analysis of previously logged
trajectories that are not deposited, and its benchmark environment, planner implementation and
twin are promised on request only. The release therefore executes on a cohort it assembles from its own configuration, at the specimen counts the article prints. Every number the
release produces is its own; the article's printed arithmetic is checked separately, against
its own tables, by a check family that is reported beside the release's and excluded from the
release's own verdict.

## Installation

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .                 # optional; the suite runs from src/ without it
```

```bash
conda env create -f environment.yml
conda activate saber
```

```bash
docker build -t saber .
docker run --rm saber python3 -m saber.cli.verify
```

## Data

The release ships no dataset. Every resource the article's Supplementary Table A1 names is
listed in `dataset_urls.txt` with the address that was fetched, the licence the row prints,
and the outcome of the fetch. The primary section holds the sixteen links whose returned
content matched the registry row; the secondary section records two GEO accessions whose
record could not be matched from a host the archive answers with an interstitial challenge.

The evaluation substrate itself is the release cohort, written by:

```bash
python3 -m saber.cli.build_cohort --out cohort --scale 1
```

`--scale` bounds how many specimens per corpus are instantiated; `0` keeps every printed
specimen (1246, 873, 964, 621 and 476 across corpora A-E). The command also writes the
registry transcription beside the cohort. Nothing in the cohort is a manuscript result: the
corpus sizes, the epoch count and the dose support come from the article, and every axis
value, observation and dose-response summary is the release's own.

## Training and evaluation

One run file selects one configuration of the seven blocks. Every reported experiment is a
`--run`:

```bash
python3 -m saber.cli.monitor    --run configs/run/primary.yaml --scale 1 --out runs
python3 -m saber.cli.estimate   --run configs/run/primary.yaml --scale 1 --out artefacts
python3 -m saber.cli.transfer   --run configs/run/primary.yaml --scale 1 --out artefacts
python3 -m saber.cli.twin       --run configs/run/primary.yaml --scale 1 --out runs
```

The acquisition comparison runs every policy in `planner.policies` (the ten rows of Table 1)
on the same feasible set, gain table and damage table, and reports each session's information,
damage, currency, exposures, budget usage, regime accuracy and saved exposures, plus the
regime-stratified frontier. The ablation configurations are the `configs/run/ablation_*.yaml`
files, one per reported variant.

The release's own read-outs are written to `runs/` and `artefacts/` by the commands above, and
its verdicts are in `verification_summary.txt`. It does not assert the article's table values,
because the trajectories they were computed from are not deposited; the article's printed
arithmetic is compared against separately, and six of those comparisons fail, which is a
finding about the article's own tables rather than about the code.

## Compute budget

The article reports 284 GPU-hours of platform training, of which 96 hours are the perception
adapters and 141 the actuation policy (Sec. 3.8); it does not report the GPU type or count, so
this release cannot state a hardware target and does not invent one. The release itself runs
on CPU: the verification pass evaluates 165 checks over a bounded cohort slice and a
48-decision session, and takes minutes rather than GPU-hours.

## Verification

```bash
python3 -m saber.cli.verify                 # default pass; opens no socket
python3 -m saber.cli.verify --probe-live    # also refetches the dataset links
```

The driver writes four artefacts at the repository root:

- `claim_to_code.json` - every mapped paper claim, its printed location, its code anchors and
  the checks that decided it, plus the deviation register;
- `verification_report.json` - every check's status with the evidence it was decided from;
- `verification_summary.txt` - the same in plain text, with the failures listed;
- `integrity_manifest.json` - one SHA-256 per shipped file and a digest of the inventory.

Statuses are `PASS`, `FAIL`, `NOT_RUN` or `BLOCKED`, decided from what ran. The two families
carry separate verdicts: `overall_status` covers everything, and `code_status` excludes the
manuscript family, because a discrepancy inside the article's own printed tables is not a
defect in the release. The shipped pass reports `code_status: PARTIALLY_VERIFIED`: nothing in
the release failed, one check is blocked because this host has no container runtime, and eleven
are not run with their reasons recorded. Running the driver replaces the artefacts, so the
committed manifest describes the tree as shipped and `scripts/verify_release.sh` runs it twice
on purpose -- the first pass lays down what the suite's own manifest test reads, and the second
pass's artefacts are the fixed point.

## Layout

```
configs/       panel, dish, perception, planner, belief, response, twin blocks and run files
scripts/       session, estimation, transfer, twin and verification launch scripts
src/saber/     spec, theory, twin, perception, belief, planner, response, evaluation, cohort,
               runtime and verification packages, plus the study orchestration module
tests/         per-module unit tests and the session smoke tests
```

## Licence

MIT; see `LICENSE`. Third-party packages and the resources the article's registry names are
listed with their licences and sources in `NOTICE`.
