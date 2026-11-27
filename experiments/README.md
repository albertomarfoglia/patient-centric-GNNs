# Experiments

Each YAML file in this folder is an **experiment spec**: it declares a grid of
runs, and `pcgnn` executes them, records everything each run produced, and never
overwrites earlier results.

## Workflow

```bash
python -m pcgnn list      experiments/mimic_rerun.yaml            # runs in the grid
python -m pcgnn folds     experiments/mimic_rerun.yaml            # write shared folds (for tabular models too)
python -m pcgnn run       experiments/mimic_rerun.yaml --gpus 0,1,2,3
python -m pcgnn status    experiments/mimic_rerun.yaml            # pending / running / done / failed
python -m pcgnn summarize experiments/mimic_rerun.yaml            # tables over folds and subsamples
```

`--filter REGEX` restricts any command to matching run ids, e.g.
`--filter 'readmission.*__full__'`. `run --dry-run` lists what would execute.
An interrupted `run` resumes: finished runs are skipped. Failed runs are skipped
unless `--retry-failed` is given.

## Terminology

* **run**: one (task, schema, pruning, subsample, variant); trains one model
  per cross-validation fold. Id: `<task>__<schema>__<pruning>[__<variant>]__s<k>`.
* **variant**: a named set of overrides of `defaults` (an ablation).

## Spec format

```yaml
name: mimic_rerun                 # results go to <output_root>/<name>/<run_id>/
description: free text
source:
  root: ../meds-to-owl-examples/MIMIC
  labels_dir: exp0-0.9            # <root>/<labels_dir>/<task>/labels/outcomes_meds_TS_<n>_<k>.joblib
  graph_dirs:                     # schema -> pruning -> <root>/<dir>/<task>/meds_<n>_<k>/*.nt
    base: {entire: exp1-full, top10: exp1-0.9}
tasks:
  - {name: readmission_30d, num_patients: 1000, subsamples: [0, 1, 2, 3, 4]}
schemas: [base]
prunings: [entire, top10]
defaults:                         # see pcgnn/config.py for every option and its meaning
  seed: 0                         # run seed = seed + subsample
  model: {num_layers: 2, num_bases: 8, embeddings: fixed_random}
  train: {lr: 0.005, patience: 50}
  features: {include_text: true, literal_mode: separate, text_mode: concat}
  evaluation: {folds: 10, split_seed: 77}
  costs: {tracking_mode: process}
variants:
  default: {}
  code_emb: {model: {embeddings: learnable_code}}
output_root: runs
cache_root: cache/preprocessed
```

Unknown keys are errors. Changing any setting changes the run's config hash; a
run directory with a different hash is never overwritten, so use a new
experiment name or variant instead.

## What a run directory contains

| File | Content |
|---|---|
| `config.yaml` | resolved settings and config hash |
| `provenance.json`, `git_diff.patch` | git commit, uncommitted changes, packages, host, GPU |
| `status.json` | state, start/end time, error and traceback if failed |
| `run.log` | log of the run |
| `graph_meta.json` | graph size and the preprocessing cache used |
| `metrics_per_fold.csv` | fold, metric, scope (class name / macro / weighted), value |
| `costs_per_fold.csv` | duration, emissions, energy, inference time, epochs, split sizes |
| `predictions.csv` | fold, patient position, true label, class probabilities |
| `emissions.csv` | raw CodeCarbon log |

## Shared folds

`folds/<task>/n<N>_s<k>_<F>fold_seed<S>.json` holds the train/validation/test
positions of every fold. The graph models and the tabular baselines
(`meds-to-owl-examples/MIMIC/run_tabular_shared_folds.py`) read the same file,
so patient assignments are identical across model families.

## Parallel runs and cost measurement

`--gpus 0,1,2,3` starts one worker process per GPU; each takes whole runs from
a queue. Each worker sees only its GPU (`CUDA_VISIBLE_DEVICES`), and CodeCarbon
in `process` mode then measures that process and that GPU only, so concurrent
runs do not inflate each other's emissions. Durations can still be affected by
contention for CPU cores; one run per GPU keeps this small.

## Validation against the first submission

`tests/test_legacy_equivalence.py` checks that, with the legacy settings
(`experiments/legacy_check.yaml`), the refactored code builds the same graph
and computes the same model outputs as the original pipeline.
