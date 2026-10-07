# Reproduce an audited result

Run commands from the repository root. Obtain datasets and weights separately under their original access conditions. No checkpoint download URL is claimed until one is actually published.

## Environment

The audited server uses Python 3.12, CUDA 12.8, PyTorch 2.7.0 and an RTX 5090. Install the compatible GPU build first:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-audit.txt
python -m unittest discover -s tests -v
```

Use an isolated environment. These commands are installation instructions; the audit did not replace the server's existing environment.

## Native data structure

```text
landslide4Sense/
  TrainData/{img/image_*.h5,mask/mask_*.h5}
  ValidData/{img/image_*.h5,mask/mask_*.h5}
  TestData/{img/image_*.h5,test/mask_*.h5}
```

Image arrays must preserve the original 14-band ordering. Data arrays are read by explicit key, with a single-key fallback; ambiguous files fail. Each image must have a mask; missing pairs are not silently skipped.

## Fixed final V8 evaluation

```bash
python -m reproducibility.evaluate \
  --data-root /path/to/landslide4Sense \
  --checkpoint /path/to/final_v8_epoch60.pth \
  --expected-sha256 9522d4755d2d9dd3e3ec6f092e06e7531c5866ab05123636e958297201070c65 \
  --channels full14 --split test --tta 1 --amp 1 --batch-size 64 \
  --out outputs/final_v8_reproduction --save-predictions
```

Only load trusted checkpoints. The original file contains optimizer metadata and therefore uses `weights_only=False`. The fixed hash verifies identity, not the safety of an arbitrary third-party checkpoint.

Outputs: `protocol.json`, `metrics.json`, `per_sample.csv`, and optionally `predictions.npz`. All plots should consume these same predictions and counts. Four-decimal targets: landslide IoU 0.4793, F1 0.6480, precision 0.6357, recall 0.6609, mIoU 0.7328. Mixed-precision GPU runs may differ by a few pixels; report actual counts and do not overwrite them with target values.

Recall lies close to a rounding boundary: the earlier retained audit has TP=163581 and FN=83950 (recall 0.6608505601, rounding to 0.6609), whereas the later audited AMP run has TP=163578 and FN=83953 (recall 0.6608384404, rounding to 0.6608). Both give IoU 0.4793 at four decimals. Retain each evaluation's own complete counts and metrics; the five displayed metrics are not guaranteed to be identical after rounding in every GPU run.

```bash
python -m reproducibility.plot_diagnostics \
  --evaluation outputs/final_v8_reproduction --out outputs/final_v8_plots
```

The plotting command checks that CSV confusion counts sum to the supplied global result. It produces a normalized confusion matrix and coverage-group summary, with empty-ground-truth patches reported separately.

## Dataset audit

```bash
python -m reproducibility.audit_dataset \
  --data-root /path/to/landslide4Sense --out outputs/dataset_audit
```

The manifest includes image and mask file hashes and image-array content hashes. It detects exact duplicates across splits, but not neighboring tiles, partial spatial overlap or related events. These need independent geospatial metadata.

## New channel correction experiment

```bash
python -m reproducibility.train \
  --data-root /path/to/landslide4Sense \
  --loveda-checkpoint /path/to/loveda_best.pth \
  --channels rgb --epochs 60 --seed 42 --max-seconds 3300 \
  --out outputs/rgb_seed42
```

For the channel comparisons, run the same command with `--channels first3_legacy` or `--channels full14` and a different output directory for each run. Each run selects on ValidData foreground IoU only. Inspect `summary.json`: `completed_epochs` must equal `planned_epochs` before describing the run as a completed 60-epoch experiment. `best.pt` is a new artifact, never a replacement for historical weights. The completed runs and their weight hashes are in [the evidence directory](../reproducibility/evidence/20261004/README.md).

The internal budget is checked between training batches. For a strict process-wide wall-clock limit on Linux, wrap the command with `timeout --signal=TERM --kill-after=15s 3420s`. A forcibly stopped process may not write a final summary; retain its logs and mark it interrupted.

Test evaluation is a separate command with the frozen `best.pt` hash. Use the matching `--channels` mode and the same TTA, AMP and batch-size options as the fixed evaluation example above. Do not select among these models on TestData. Report every arm, including the post-hoc full14 extension; these reused benchmark results do not become a new independent holdout.

## Prespecified three-seed comparisons and regional transfer

See [the complete dated protocol](PROSPECTIVE_PROTOCOL_20261004.md) before running this suite. It uses 18 Landslide4Sense fits and three small external fine-tunes. The portable orchestration wrapper implements the same arm order and freeze-before-evaluation rule as the archived server runner. The dated run's actual source snapshots and hashes remain the record of execution.

On Linux, first run all matched comparisons. Substitute your own dataset and checkpoint locations; output directories must be new.

```bash
python -m reproducibility.run_matched_suite \
  --data-root /path/to/landslide4Sense \
  --loveda-checkpoint /path/to/loveda_best.pth \
  --expected-source-sha256 42c8a2967afe258bc8cc5cabc774e3cc5caefa66e60d62fcc9b8751424eca319 \
  --out outputs/matched_suite --training-budget-seconds 14400
```

The suite uses seeds 2026, 2027, and 2028. TrainData-only checkpoints are selected on validation landslide IoU; TrainVal refits use fixed epoch 60. A failed or time-limited run blocks the test phase. The original audit counted 1,034.951 seconds of earlier training toward its four-hour limit; a fresh reproduction starts with zero prior seconds unless `--prior-training-seconds` is supplied.

### Restore the exact regional inputs

Recover and audit Chimanimani using the commands in [the source-data audit](../reproducibility/evidence/sen12_20261004/README.md). The regional root must contain `raw/` and `audit_v2/sample_manifest.csv`; use `audit_v2` as the output of `reproducibility.audit_sen12_subset`.

The Dominica holdout uses the 208 files enumerated in the dated `extra_region_inventory.json`. Extract them from the pinned third archive:

```bash
python -m reproducibility.extract_frozen_region \
  --archive /path/to/s2_part03.tar.gz \
  --inventory q2_experiments/runs/prospective_20261004/extra_region_inventory.json \
  --expected-archive-sha256 622ea1eccefd49072170e2c341c9cdac049e2ab2fd135eebb02eb7afb42414c1 \
  --out outputs/dominica_raw

python -m reproducibility.prepare_external \
  --chimanimani-root /path/to/chimanimani_recovery \
  --dominica-root outputs/dominica_raw/raw \
  --dominica-manifest q2_experiments/runs/prospective_20261004/extra_region_inventory.json \
  --out outputs/prepared_external
```

Preparation freezes the temporal/quality rule, channel names, 40/10 support split, test IDs, and display IDs. It must finish before adaptation. The holdout is an archive-defined regional subset, not the official full Sen12 benchmark. The new prepared-file hashes and protocol creation time belong to the reproduction run; they need not match timestamps in the original run's JSON.

### Adapt, evaluate, and verify

```bash
python -m reproducibility.external_experiment \
  --data-root outputs/prepared_external \
  --matched-root outputs/matched_suite \
  --out outputs/matched_suite/external_experiment

python -m reproducibility.verify_prospective \
  --source outputs/matched_suite --prepared outputs/prepared_external \
  --out outputs/matched_suite/integrity_checks.json

python -m reproducibility.report_prospective \
  --source outputs/matched_suite --prepared outputs/prepared_external \
  --out outputs/review
```

`external_experiment` uses the remaining part of the four-hour cumulative training allowance. The direct RGB source models are fixed by seed; no benchmark score chooses among them. Three Chimanimani fine-tunes complete before Dominica test predictions are generated. Existing run directories are never resumed or overwritten automatically.

The review folder contains CSVs and PNG/PDF/SVG figures. Qualitative plots require the saved prediction archives and the eight input examples produced by evaluation. Raw inputs and model weights are not bundled in the public repository. Metadata-only verification can check published metrics and manifests, but cannot independently regenerate predictions without the datasets and corresponding weights or a new training run.
