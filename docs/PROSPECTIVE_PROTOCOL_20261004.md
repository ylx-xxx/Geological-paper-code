# Frozen comparisons and regional transfer protocol

This document describes the experiment registered locally on 4 October 2026. It accompanies the machine-readable plans, sample manifests, training histories, and evaluation outputs. Results are added only after the complete prescribed model set has been evaluated. This is a new experiment; it does not reconstruct the historical Sen12 40/10/210 split.

## Matched Landslide4Sense comparisons

The same three seeds (2026, 2027, and 2028) are used in each of six configurations. All runs use the same architecture, optimizer, normalization, augmentation, loss, and 60-epoch cosine schedule.

| Configuration | Input | Initialization | Training samples | Checkpoint rule |
|---|---|---|---:|---|
| RGB / LoveDA | B4, B3, B2 | LoveDA | 3,799 | Maximum validation foreground IoU; earliest tie |
| Full14 / mean extras | 12 spectral channels, slope, DEM | LoveDA; mean extra kernels | 3,799 | Same validation rule |
| Full14 / random extras | Same 14 channels | LoveDA; random extra kernels | 3,799 | Same validation rule |
| Full14 / scratch | Same 14 channels | Random throughout | 3,799 | Same validation rule |
| TrainVal / mean extras | Same 14 channels | LoveDA; mean extra kernels | 4,044 | Fixed epoch 60 |
| TrainVal / random extras | Same 14 channels | LoveDA; random extra kernels | 4,044 | Fixed epoch 60 |

The TrainData-only arms use 245 ValidData samples for selection. Fixed TrainVal refits concatenate the native 3,799 training and 245 validation pairs and use every sample for training. They do not select an epoch using a held-out validation set. Original checkpoint directories remain unchanged.

For each run, the source checkpoint identity, source-code hashes, actual command, seed, sample manifest, class weights, epoch history, and selected checkpoint hash are saved. The 18 model identities are frozen together before any of these models are evaluated on benchmark TestData. All 18 scores are reported. No test-based promotion is performed.

This procedure prevents test-based selection within this new suite. It does not undo historical use of Landslide4Sense TestData feedback. The existing benchmark should continue to be described as previously exposed during development.

Dataset identity, image-quality, label-count, spatial-footprint, and exact-duplicate audits occur before model evaluation. “Test opening” here refers to model inference and score inspection, not the first read of test files. The archived plan's `no_test_access_until_all_training_complete` field is an imprecise legacy name for this model-evaluation gate; it must not be interpreted as an assertion that no test files or annotations were inspected for dataset auditing.

### Shared recipe

- AdamW; initial learning rate 0.00003; weight decay 0.05; cosine decay to 0.000001.
- Batch size 64; 60 epochs; no dropped final training batch.
- Per-patch, per-channel normalization: nonfinite values become zero, values are clipped to P1/P99, and the clipped data are standardized. A standard deviation below 0.000001 uses denominator one.
- Random horizontal and vertical flips, 90-degree rotations, and Gaussian noise with standard deviation 0.02, following the stored implementation.
- Weighted cross-entropy plus weighted Dice, each with coefficient one; cross-entropy label smoothing 0.02.
- Class weights use training-pixel frequencies only: inverse square root with epsilon 0.000000000001, division by their mean, and clipping to [0.5, 2.5].
- Three-view TTA: original, horizontally flipped, and vertically flipped inputs. Inverse-aligned logits are averaged and the class with the largest logit is selected.
- AMP and GPU arithmetic are recorded. Bitwise agreement across hardware is not claimed.

The full14 transfer copies the source RGB kernel positions to the first three stored target channels and initializes the remaining channels by the assigned rule. It is therefore a comparison of complete channel-and-initialization configurations. It does not isolate spectral information while holding a physically aligned transfer map constant.

Compatible encoder and decoder parameters are transferred together, while the seven-class classifier is replaced by a binary classifier. The new loader also resizes the two mismatched final-stage relative-position bias grids by bicubic interpolation. Historical loading skipped those tensors. The matched suite therefore follows an explicit new loading recipe and is not an exact replay of every historical optimization trajectory.

## New Sen12 regional experiment

### Source and support

The three RGB / LoveDA models above supply the source checkpoints. For each seed, a copy is fine-tuned on 40 Chimanimani patches and selected using 10 separate Chimanimani validation patches. The direct source model and adapted model are both evaluated on the same Dominica Maria holdout.

This tests physical-RGB transfer from Landslide4Sense and adaptation using another region. It is not a direct external evaluation of the original 14-channel V8 model or of the new full14 arms. Sen12 does not provide the identical 14-channel input used in Landslide4Sense.

### Data identity and footprint

The Raw/Sentinel-2 dataset is pinned to Hugging Face revision `311f426d0e2fa9b532772fbee0641e2f9db6da00`. Archive and extracted-file SHA256 checks identify the downloaded data. The recovered pool includes 1,133 Chimanimani files and all 208 Dominica Maria files present in `s2_part03.tar.gz`.

The Dominica pool is defined by this archive subset. No claim is made that it covers every Dominica sample in the full 28-archive release. China files in the same archive are excluded because their multi-year, mixed-event annotation dates do not fit this single-event temporal protocol. The official LD candidate list is reference evidence; this custom region-held-out split is not the official benchmark split.

### Temporal and quality policy

- Read physical RGB by variable name: B04, B03, B02.
- Convert the stored `time,x,y` axes to `C,y,x` for the image and the same spatial orientation for the mask.
- Use dataset event-date anchors: Chimanimani 2019-03-15 and Dominica Maria 2017-09-23. These are metadata anchors, not independently verified hurricane landfall dates.
- Among acquisitions strictly after the anchor and at most 180 days later, choose the image with the greatest valid-pixel fraction; use the earliest acquisition for ties. The annotation mask is not used to choose the date.
- Valid SCL classes are 2, 4, 5, 6, and 7. Require at least 80% valid pixels. Ignore other pixels in the loss and evaluation by assigning target index 255.
- Retain pure-background patches. Use the same normalization formula as the RGB Landslide4Sense branch.

All 1,133 Chimanimani files satisfy this policy. Four of the 208 Dominica files fail the image-quality criterion. The fixed test set therefore contains 204 patches: 93 with retained foreground and 111 with no raw foreground annotation.

Here, background means zero in the released inventory mask; it is not an independent field verification of absence. The original dataset construction includes sampled non-annotated patches, so the observed foreground/background mixture does not estimate whole-region prevalence. The official LD task uses positive patches only; our retained-background regional split is a distinct evaluation protocol. See the [data construction paper](https://www.nature.com/articles/s41597-025-06167-2) and [official task definitions](https://github.com/PaulH97/Sen12Landslides).

### Spatial support selection

Chimanimani is divided into western training and eastern validation pools with a prescribed minimum 1,280 m gap between tile footprints. A label-blind draw with seed 20261004 selects 40 and 10 patches. The realized minimum training-to-validation footprint distance is 2,862.167 m. The draw is not repeated to increase the number of positive examples.

| Split | Patches | Patches with retained foreground | Valid pixels | Foreground pixels |
|---|---:|---:|---:|---:|
| Chimanimani training | 40 | 9 | 655,195 | 1,637 |
| Chimanimani validation | 10 | 3 | 163,762 | 2,628 |
| Dominica test | 204 | 93 | 3,318,781 | 41,094 |

The 50-patch annotation budget means 40 training plus 10 validation patches. It does not mean 50 patches provide training gradients. This experiment uses one support draw and three optimization seeds; it does not estimate variation over different support draws.

### Fine-tuning and test opening

Each source model is fine-tuned for 40 epochs using batch size eight and the shared optimizer, augmentation, and loss recipe. Class weights are computed from the 40 training masks. The checkpoint with maximum validation landslide IoU is selected, with the earliest epoch retained for a tie. All three adapted checkpoints and all three direct source checkpoints are frozen before test evaluation begins. The output directory is new and a test-opening marker is created with exclusive file access.

Both stages use the same test manifest, valid-pixel mask, normalization, and three-view TTA. The code saves every per-sample TP, FP, FN, and TN count, global metrics, full prediction masks, and checkpoint identity. A separate verifier recomputes global metrics from the saved per-sample counts and checks the recorded selection rule and manifest identities.

## Uncertainty and figures

Report the mean and sample standard deviation across the three seeds and retain every individual result. Direct-to-adapted differences are paired by seed.

Within the single Dominica event, the protocol groups tile centers into a fixed 2,560 m grid and resamples those spatial blocks 2,000 times for the foreground-IoU interval. This is conditional spatial uncertainty within one event. It is not an estimate of uncertainty across independent events; correlations may extend beyond the chosen block size. Do not treat pixels or adjacent patches as independent replicates.

Eight display IDs are fixed before inference: two pure-background samples and six samples spaced across foreground-coverage ranks. Seed 2026 supplies the qualitative figures. Predictions, error overlays, and displayed metrics all come from the saved evaluation output. Background-only examples show false-positive rate rather than foreground IoU. The plotting code verifies displayed pixel counts against the CSV.

## Limits that remain

1. Historical benchmark feedback remains part of the development history. The new benchmark suite does not make that TestData untouched.
2. The author confirms no prior training in Sen12 regions outside Chimanimani. A complete audit of every historical evaluation is unavailable; absence of files on the current server is not proof of absence of prior access.
3. The current adaptation excludes Dominica completely. Exact image-array checks against all 4,844 Landslide4Sense patches, including rotations and reflections, found no matches. These checks cannot rule out neighboring footprints, partial overlaps, or altered radiometry. Full Landslide4Sense validation/test georeferencing is not publicly provided.
4. The test contains one event and an archive-defined footprint. It cannot support claims of generalization across many independent events or complete Sen12 benchmark performance.
5. The small support set contains few positive pixels. The three optimization seeds do not resolve uncertainty from the support selection itself.
6. The historical external result and split remain unverified. New results must have a distinct name, split description, and sample count.
7. The random-initialization arm compares the full retained LoveDA initialization against training from scratch. This suite does not include an ImageNet-only control. The historical LoveDA implementation supports loading ImageNet weights, but its retained checkpoint does not contain the original command or initialization arguments. The specific added value of the LoveDA stage over ImageNet alone is not isolated by these comparisons.
8. Choosing the clearest image within 180 days permits delayed observations. This retrospective transfer experiment does not measure immediate post-disaster response latency or establish real-time mapping performance.

## Primary data documentation

- [Landslide4Sense repository and band-order FAQ](https://github.com/iarai/Landslide4Sense-2022)
- [Sen12Landslides repository and data loaders](https://github.com/PaulH97/Sen12Landslides)
- [Sen12Landslides data paper](https://www.nature.com/articles/s41597-025-06167-2)
- [Pinned Raw/Sentinel-2 release](https://huggingface.co/datasets/paulhoehn/Sen12Landslides/tree/311f426d0e2fa9b532772fbee0641e2f9db6da00)
