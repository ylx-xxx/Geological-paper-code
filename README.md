# LoveDA → Landslide4Sense: landslide segmentation

Research code for cross-domain remote sensing pretraining with Swin-UPerNet. LoveDA supplies source-domain land-cover supervision. Landslide4Sense provides multispectral images, topographic inputs, and binary landslide masks.

**Start here:** [Reproduce results](docs/REPRODUCE.md) · [Experimental protocol and limitations](docs/EXPERIMENT_PROTOCOL.md) · [Artifact status](docs/ARTIFACT_STATUS.md)

## Evidence status

The retained final model reports landslide IoU **0.4793**, F1 **0.6480**, precision **0.6357**, recall **0.6609**, and mIoU **0.7328** on benchmark TestData after TrainData + ValidData training. This is a reproducible benchmark result, not a state-of-the-art claim. Historical test-informed candidate comparisons limit claims of untouched test evaluation.

TrainData-only Swin-UPerNet scores are 0.3916 without LoveDA and 0.4426 with LoveDA; U-Net reaches 0.4460. These must be separated from the final model trained with additional data. The separate full14 ablation (0.4231) uses a different schedule and selection criterion.

**Channel correction:** historical `rgb` ablations use `[0,1,2]`. Native Landslide4Sense physical RGB uses `[3,2,1]`. New code distinguishes `first3_legacy` and `rgb`; historical scores are not silently reassigned.

The completed seed-42 correction experiments used TrainData only, 60 epochs and validation foreground-IoU selection. Test IoU was **0.3846** for the first three stored bands, **0.4548** for physical RGB and **0.4231** for full14. The full14 arm was added after inspecting the paired results. These results do not support a general claim that all 14 channels outperform physical RGB. See the [complete evidence and qualifications](reproducibility/evidence/20261004/README.md).

## Repository map

| Directory | Purpose |
|---|---|
| `reproducibility/` | Shared normalization, explicit channel modes, strict evaluation, dataset audit and validation-only training |
| `reproducibility/evidence/` | Audited metrics, source/weight identities and experiment status |
| `tests/` | Scientific invariants: normalization, pooled metrics, channels, ignored labels and selection split |
| `train_la_qz/` | Historical LoveDA pretraining |
| `train_l4s_qz/` | Historical target architecture and training |
| `train_l4s_qz_v*/` | Historical variants retained for traceability |
| `q2_experiments/` | Manuscript tables, baseline outputs and figures; consult artifact status before reuse |
| `q2_extra_ablation/` | Historical channel/loss/initialization ablations and exploratory external outputs |

## Quick verification

```bash
python -m unittest discover -s tests -v
python -m reproducibility.evaluate --help
python -m reproducibility.train --help
```

See [REPRODUCE.md](docs/REPRODUCE.md) for environment installation, data layout, fixed hashes and full commands. The verified original checkpoint and third-party data are not bundled in this repository. New outputs must use a fresh directory.

## Framework

![Overall framework](流程图.png)

## External experiment status

The new frozen experiment is complete: **18 matched Landslide4Sense fits, three regional fine-tunes, and 24 frozen-model evaluations**. See the [protocol](docs/PROSPECTIVE_PROTOCOL_20261004.md) and [small CSV result tables](q2_experiments/tables/prospective_20261004/). The full run records, model binaries, prediction masks, and candidate figures are retained outside Git and identified by the run manifests.

| New setting | Landslide IoU, mean ± sample SD over three seeds |
|---|---:|
| TrainData only, physical RGB / LoveDA | 0.4492 ± 0.0031 |
| TrainData only, full14 / LoveDA, mean extra kernels | 0.4233 ± 0.0195 |
| TrainData only, full14 / LoveDA, random extra kernels | 0.4386 ± 0.0050 |
| TrainData only, full14 / scratch | 0.3877 ± 0.0022 |
| TrainData + ValidData, full14 / mean extras, fixed epoch 60 | 0.4496 ± 0.0039 |
| TrainData + ValidData, full14 / random extras, fixed epoch 60 | 0.4759 ± 0.0110 |
| Dominica, direct transfer of the physical-RGB models | 0.0031 ± 0.0011 |
| Dominica, after Chimanimani adaptation | 0.0353 ± 0.0068 |

The external protocol uses 40 Chimanimani training and 10 validation patches, with 204 Dominica test patches from a pinned archive subset. All three seeds improve after adaptation, but absolute accuracy remains very low. These results **do not demonstrate strong cross-region generalization**. This is a physical-RGB transfer experiment, not a direct external test of the original 14-channel V8 model. One event, one support draw, sampled background, delayed image acquisition, and incomplete cross-dataset georeferencing limit interpretation.

Historical Sen12 figures remain under `q2_extra_ablation/external_sen12_rgb/`. The recovered 1,133 Chimanimani files restore source data, but the old 40/10/210 IDs, generating scripts, and matching checkpoint remain unavailable. Those scores remain unverified exploratory evidence; the new experiment uses a different, fully recorded protocol. See the [source-data audit](reproducibility/evidence/sen12_20261004/README.md).

## 中文说明

本仓库区分历史实验和新实验。新增 18 组统一配方对照、3 次外部微调、逐样本 CSV 和候选图片均已核验。外部 IoU 均值从 0.0031 升至 0.0353，但精度仍低，不能据此声称较强泛化。旧 0.4793、0.4426、0.4231 和旧 Sen12 结果保持各自来源，不混用记录。论文正文和原配图等待作者验收后才修改；代码整理与实验完成不保证期刊录用。

## Data and references

- [Landslide4Sense official dataset and baseline](https://github.com/iarai/Landslide4Sense-2022)
- [LoveDA official repository](https://github.com/Junjue-Wang/LoveDA)
- [Swin Transformer](https://doi.org/10.1109/ICCV48922.2021.00986)
- [UPerNet](https://doi.org/10.1007/978-3-030-01228-1_26)

Please respect the original dataset and dependency licenses. No new license grant for third-party assets is implied.
