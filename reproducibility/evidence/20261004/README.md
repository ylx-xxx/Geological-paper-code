# Server verification — 2026-10-04

The four archived checkpoints were loaded strictly and evaluated on all 800 native TestData patches with identical per-band normalization, three-view logit TTA, batch size 64 and mixed precision.

| Checkpoint / experiment | Landslide IoU | F1 | Interpretation |
|---|---:|---:|---|
| Final V8 epoch 60 | 0.47931434 | 0.64802230 | Reproduces manuscript rounding 0.4793 / 0.6480 |
| Main TrainData-only, mIoU-selected | 0.44260641 | 0.61362046 | Reproduces archived main-comparison CSV |
| Full14 ablation, foreground-IoU-selected | 0.42309529 | 0.59461273 | Reproduces archived ablation score |
| Legacy first-three-band ablation | 0.39830069 | 0.56969248 | Not physical RGB |

The final V8 SHA256 remains `9522d4755d2d9dd3e3ec6f092e06e7531c5866ab05123636e958297201070c65`. Compared with the 2026-10-03 mixed-precision audit, the current final confusion matrix has three fewer true positives and one fewer false positive. The actual counts are preserved in `final_v8/metrics.json`; no target values were substituted. All five headline metrics retain their manuscript four-decimal values. CUDA mixed-precision evaluation is not claimed to be bitwise deterministic.

`dataset/manifest.csv` covers 3,799 training, 245 validation and 800 test patches. Exact image-array hashes do not overlap across splits. Empty-ground-truth patch counts are 1,568 / 99 / 264. This content audit does not establish spatial or event independence.

`tests.json` records nine successful tests on the server. The local Windows environment lacks `timm`, so the model-dependent test requires the documented environment there. No dependency installation was forced into that environment.

`inspection.json` contains stored checkpoint metadata and observed package versions. Missing historical command-line arguments must not be reconstructed as known facts from current script defaults.

The external Sen12 dataset and generating scripts were not found on the server. The historical external score remains unverified in this audit.

## Completed paired channel experiment

Both runs completed 60 epochs with seed 42. Checkpoints were selected on validation foreground IoU before the two test results were computed. Total paired training/evaluation pipeline time was about 701 seconds.

| Input | Best validation epoch | Validation IoU | Test IoU | Test F1 | Test precision | Test recall |
|---|---:|---:|---:|---:|---:|---:|
| First three stored bands | 19 | 0.3904 | 0.3846 | 0.5555 | 0.6434 | 0.4887 |
| Physical RGB, B4/B3/B2 | 18 | 0.4619 | 0.4548 | 0.6252 | 0.5520 | 0.7209 |

The single-seed paired result shows that the channel definition materially changes performance in this setting. It is not evidence of multi-seed statistical significance. The new runs use explicit relative-position adaptation and a cached input pipeline; old and new optimization trajectories are not identical. Both results are retained, with no test-based promotion of the final model.

## Completed full14 extension

After inspecting the two paired test results, a full14 comparator was added using the same seed, data, 60-epoch schedule, loss, augmentation, positional-bias resizing and validation selection. The timing of this decision is recorded in `full14_extension_plan.json`; it was not a preregistered third arm. The run completed all 60 epochs and selected epoch 18 at validation IoU 0.478535.

| Input | Best validation epoch | Validation IoU | Test IoU | Test F1 | Test precision | Test recall |
|---|---:|---:|---:|---:|---:|---:|
| All 14 stored channels | 18 | 0.4785 | 0.4231 | 0.5946 | 0.5795 | 0.6105 |

Physical RGB exceeds full14 on this test evaluation (0.4548 versus 0.4231), although full14 has the higher validation IoU. The result does not support a blanket claim that adding multispectral and topographic channels improves performance. Full14 also uses positional source-kernel copying with mean initialization for extra channels, not spectral alignment; channel choice and transfer initialization are therefore coupled. This is one seed on a reused benchmark, and no test-based model replacement or additional parameter search was performed.

The three runs have identical training and validation sample manifests. Their saved epochs match the maxima of their respective validation histories. All test CSV confusion counts close exactly to their global metrics. Their combined recorded run time, including each run's setup and validation, is 1,034.95 seconds (17.25 minutes), below the authorized two-hour training budget.

| New checkpoint | SHA256 |
|---|---|
| First three bands | `723d8baa1b536b3c85628d81f8d1309e79a3d21cbea377e1ec2a7aa5361d460f` |
| Physical RGB | `c00138393cf96b03a929a2a6fdfa37c1861ad1d4522388c52d50188bc7acc587` |
| Full14 | `2e5e4bf1180e346046b4aa1a54b2f6ad067ac969d8a7989d75177a26435a7e25` |

Weight binaries are backed up separately and are not included in Git. The historical final V8 remains unchanged; its 0.4793 score uses TrainData + ValidData and is not a same-data comparison against these three runs.
