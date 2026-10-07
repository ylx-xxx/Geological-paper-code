# Experiment protocol and evidence boundaries

## Historical experiments

| Result | Training data | Selection / evidence |
|---|---|---|
| Swin-UPerNet + LoveDA, foreground IoU 0.4426 | TrainData | Stored `l4s_best.pth`, epoch 48; scheduler T_max=100; selected by validation mIoU |
| Full14 input ablation, foreground IoU 0.4231 | TrainData | Stored `best.pth`, epoch 15; scheduler T_max=60; selected by validation landslide IoU |
| Final V8, foreground IoU 0.4793 | TrainData + ValidData | Retained epoch-60 checkpoint; subsequent test-informed comparisons limit evaluation independence |

The main comparison and the full14 ablation are different recipes, not duplicate measurements of the same controlled baseline. Source inspection also finds `drop_last=True` in the main training loader and `False` in the ablation loader. Automatic class-weight code differs (clipping caps 5.0 versus 2.5 and different numerical epsilons); neither upper cap should bind for normalized two-class inverse-square-root weights. These are recipe differences, not evidence that each difference caused a score change. Complete historical command-line arguments were not saved for the main run.

The original main training code can update `best_landslide_iou` inside the mIoU branch before checking the foreground-specific save condition. Consequently, the separately named foreground-best file is not a reliable proof of the best foreground epoch. The audited main 0.4426 result maps to the mIoU-selected file. New training has one explicit validation foreground-IoU selection rule.

Historical fine-tuning scripts used TestData metrics to compare/promote candidates. Seven available records did not promote a replacement. Retaining the checkpoint does not prove that TestData was never consulted. Reproducing a benchmark score establishes computational consistency, not prospective evaluation independence. Historical artifacts are preserved for traceability.

## Channel definitions

According to the [official IARAI dataset description](https://github.com/iarai/Landslide4Sense-2022), stored channels are Sentinel-2 B1–B12 (B8A omitted), slope, and DEM. Under that ordering:

- `rgb`: zero-based `[3, 2, 1]`, corresponding to B4/B3/B2.
- `first3_legacy`: `[0, 1, 2]`, corresponding to B1/B2/B3.
- `rgb_topo`: `[3, 2, 1, 12, 13]`.
- `full14`: `[0, ..., 13]` in native order.

Historical ablation scripts called the first three channels `rgb`. Their scores remain historical first-three-band results. Changing inference channels for an already trained checkpoint would change its input distribution; it would not repair the original training. Use the new explicit channel modes for new training.

The inspected native HDF5 sample has shape `(128,128,14)` and no band-name attributes. Semantic band ordering relies on dataset provenance, not inference from pixel values. Georeferencing is not present in that sample; exact content hashes alone cannot prove spatial independence.

## New training and evaluation

`reproducibility.train` accepts TrainData and ValidData only. It records the source checkpoint hash, channel indices, code hashes, sample manifests, losses, seed, optimizer recipe, and selection criterion. It never evaluates TestData during training. A new output directory is required. Existing checkpoints are read-only inputs.

The source and target image sizes produce two mismatched relative-position bias grids in the last encoder stage. Historical transfer skips these tensors. New training explicitly resizes the grids by bicubic interpolation and rejects any other missing non-classifier weights. This follows the standard resizing approach in [timm](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed_rel.py). All three new channel runs use the same adaptation; their results are not exact historical training replays.

The paired correction experiment uses the same seed, 60-epoch schedule, loss, data, augmentation, and validation selection for `first3_legacy` and physical `rgb`. Normalized inputs are cached in RAM. This changes data loading and random-number scheduling relative to historical scripts, so the two new runs should be compared with each other, not treated as exact replays of old optimization trajectories. CUDA bitwise determinism is not guaranteed. Runs stopped by the time budget must be labelled incomplete.

After inspecting the paired test results, a full14 arm was added with the same recipe. All three runs completed 60 epochs. Test foreground IoU was 0.3846 / 0.4548 / 0.4231 for first3 / physical RGB / full14. This is a single-seed comparison on a reused benchmark, with a post-hoc full14 extension. It does not establish statistical significance or a universal benefit from additional channels. For full14, source patch-embedding kernels are copied by position and extra channels use their mean; this is not a spectrally aligned transfer. The experiment therefore compares complete channel/initialization configurations, rather than isolating information content alone. The final 0.4793 model also uses additional training data and cannot resolve this comparison.

`reproducibility.evaluate` requires the expected checkpoint SHA256 and strict state loading. It pools confusion counts over all pixels and also exports per-sample scores. Empty-ground-truth patches are excluded from foreground patch means, with their false positives retained in global metrics. It does not search thresholds or promote models. Frozen benchmark evaluations remain benchmark reuse; new spatial/event-level data are required for a genuinely untouched holdout.

## Sen12 external experiment

The initial inspection found historical metric tables, a learning-curve CSV and figures, but no Sen12 dataset, generating training/evaluation script or split manifest in the inspected server locations. The curve's confusion counts cover ten 128×128 patches; this is consistent with a ten-patch validation set but does not identify those patches. The CSV's `test_miou` label cannot settle this question.

Subsequent local recovery verified three official Raw/Sentinel-2 archives and extracted 1,133 Chimanimani files. They include all 349 regional candidates in the pinned official LD split; the historical 40/10/210 split is still unavailable. The archive and per-file identities, label-count agreement and quality findings are recorded in [the recovery evidence](../reproducibility/evidence/sen12_20261004/README.md). These newly recovered source files do not establish which subset or preprocessing produced the old external score.

The historical label budget is 40 training + 10 validation patches, with 210 reported evaluation patches from a positive-only Chimanimani subset. The original sample IDs, channel mapping, spatial overlap, and checkpoint selection must be recovered before this can support independent external validation. No external result is silently regenerated or relabelled.

## Submission-critical remaining evidence

The [4 October experiment](PROSPECTIVE_PROTOCOL_20261004.md) now supplies 18 matched fits, three external fine-tunes, 24 evaluations, saved identities and manifests, three-seed summaries, and spatial-block uncertainty within one test event. The [complete results](../q2_experiments/runs/prospective_20261004/README.md) reconcile global counts with per-sample records. All 1,224 saved external predictions were separately checked against those records.

The new RGB regional transfer improves mean IoU from 0.0031 to 0.0353 after Chimanimani adaptation. Its low absolute accuracy limits a positive generalization claim. This is not a recreation of the historical external experiment. New TrainVal runs use fixed epoch 60; the earlier statement about validation selection applies to TrainData-only runs.

Remaining limits:

1. Historical TestData feedback must remain disclosed. New disciplined evaluation does not restore its original independence.
2. The old external 40/10/210 split and checkpoint remain unverified. Results from the new 40/10/204 regional protocol cannot inherit that claim or be directly compared as a replication.
3. One event and one support draw do not establish multi-event transfer. Exact-array duplicate checks cannot exclude partial spatial overlap with incompletely georeferenced source data.
4. The full14 transfer uses positional kernel copying, while RGB uses physical B4/B3/B2. Input and initialization effects are coupled. No ImageNet-only control isolates the added contribution of the LoveDA stage.
5. Further method development must use development regions. Once these Dominica scores have been inspected, subsequent tuning cannot retain the same claim of a newly untouched test region.
6. Author-confirmed availability, funding, contribution, and competing-interest statements, and all manuscript edits, remain for the author's review.

This repository supports reproducibility review. It does not certify a journal quartile or guarantee acceptance.
