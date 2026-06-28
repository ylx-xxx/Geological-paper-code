# Geological Paper Code (地质论文代码)

This repository contains code and figures for our geological paper experiments.  
（本仓库包含地质论文实验的代码和图表。）

## Repository Structure (文件夹说明)

This repository is organized according to the experimental pipeline of the geological remote-sensing segmentation paper. The main folders are described below so that readers can quickly understand the role of each part.  
（本仓库按照地质遥感分割论文的实验流程进行组织。下面对主要文件夹进行说明，便于读者快速理解每个部分的作用。）

| Folder / File | Description |
| --- | --- |
| `train_la_qz/` | LoveDA pre-training and semantic segmentation module. This folder contains the LoveDA dataset loader, Swin-UPerNet model, loss functions, metrics, utilities, and training script. It is mainly used for 7-class remote-sensing semantic segmentation, including Building, Road, Water, Barren, Forest, Agricultural, and Background. <br>（LoveDA 预训练与语义分割模块，包含数据读取、Swin-UPerNet 模型、损失函数、评价指标、工具函数和训练入口，主要用于 7 类遥感语义分割任务。） |
| `train_l4s_qz/` | Landslide4Sense fine-tuning and landslide segmentation module. This folder contains the 14-band H5 data loader, binary landslide segmentation model, LoveDA checkpoint transfer logic, automatic class weighting, training, validation, and metric recording code. <br>（Landslide4Sense 微调与滑坡识别模块，包含 14 波段 H5 数据读取、二分类滑坡分割模型、LoveDA 权重迁移、自适应类别权重、训练验证和指标记录代码。） |
| `q2_extra_ablation/` | Extra experiments, ablation studies, and external dataset evaluation module. It stores additional experimental results, including zero-shot and few-shot prediction visualizations on the external SEN12-RGB dataset. <br>（额外实验、消融实验与外部数据集评估模块，用于保存补充实验结果，包括 SEN12-RGB 外部数据集上的零样本和少样本预测可视化结果。） |
| `q2_extra_ablation/external_sen12_rgb/figures/` | Visualization results for SEN12-RGB external evaluation. The figures in this folder are used to show qualitative prediction examples in the README and paper-related materials. <br>（SEN12-RGB 外部评估的可视化结果目录，用于展示 README 和论文材料中的定性预测示例。） |
| `流程图.png` | Overall experimental workflow figure. It illustrates the main pipeline of the geological remote-sensing segmentation experiments. <br>（总体实验流程图，用于展示地质遥感分割实验的整体流程。） |
| `README.md` | Project description file. It explains the repository purpose, folder structure, and example prediction figures. <br>（项目说明文件，用于介绍仓库用途、文件夹结构和预测示例图。） |

## Overall Experimental Workflow (总体实验流程图)

![Overall experimental workflow](./流程图.png)  
（总体实验流程图）

## Prediction Examples on SEN12-RGB (SEN12-RGB 数据集上的预测示例)

We also evaluate our model on the external SEN12-RGB dataset under zero-shot and few-shot settings. Below are example predictions.  
（我们还在外部 SEN12-RGB 数据集上评估了模型在零样本和少样本设置下的表现。以下是部分预测结果。）

### Zero-Shot Predictions (零样本预测)
![Zero-shot predictions](./q2_extra_ablation/external_sen12_rgb/figures/sen12_rgb_zeroshot_predictions.png)

### Few-Shot (50 shots) Predictions (少样本（50张）预测)
![Few-shot predictions](./q2_extra_ablation/external_sen12_rgb/figures/sen12_rgb_fewshot50_predictions.png)
