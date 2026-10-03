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
