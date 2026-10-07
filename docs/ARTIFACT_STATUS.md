# Artifact status

- `train_la_qz/`, `train_l4s_qz/`, `train_l4s_qz_v*/`: historical implementations. They preserve the development history. Start with `reproducibility/` for audited evaluation and new channel-corrected training.
- `q2_experiments/tables/`: historical manuscript tables; training protocols differ across rows. See `EXPERIMENT_PROTOCOL.md`.
- `q2_extra_ablation/runs/`: archived ablation logs. Historical `rgb` means first three stored bands, not physical RGB.
- `q2_experiments/figures/visualizations/final_v8/`, `figures/area_group/` and `figures/sample_distribution/`: legacy diagnostic outputs generated with preprocessing inconsistent with the final evaluation. Do not use these to support final-model sample-level conclusions.
- `q2_experiments/figures/verified_final_v8_20261003/`: corrected diagnostic figures from the previous checkpoint/normalization audit. Global and per-sample counts reconcile in that audit.
- `reproducibility/evidence/20261004/`: current executable-code tests, frozen checkpoint evaluations, data manifest audit and prospective run evidence. Check each file's status; incomplete runs are not finished experiments.
- `q2_extra_ablation/external_sen12_rgb/`: historical exploratory figures/tables. Source scripts, custom split identity and matching checkpoint remain unverified; these scores are not independently reproduced evidence.
- `reproducibility/evidence/sen12_20261004/`: local Raw/Sentinel-2 data recovery, covering 1,133 regional files and all 349 pinned official LD candidates. This audit restores source data, not the historical custom experiment.

- `q2_experiments/runs/prospective_20261004/`: completed matched and regional experiments retained on the data disk and in a private local review package. The full histories, per-sample counts, model hashes, executed source snapshots, and prediction checks are not part of the slim public Git checkout.
- `q2_experiments/tables/prospective_20261004/`: new three-seed results, configuration and historical evidence tables. They do not overwrite the historical manuscript tables.
- `q2_experiments/figures/prospective_20261004/`: eight locally retained candidate figures in PNG, PDF, and SVG. Qualitative sample IDs were fixed before inference; poor cases are retained. Figure inclusion in the manuscript awaits author approval.

The complete evidence was downloaded locally, including all six models' external prediction masks and the eight displayed inputs. Raw datasets and NPZ arrays are not published. New checkpoint binaries remain on the server data disk; they are identified by SHA256 but are not included in this repository. Public result tables provide summary verification; full per-sample and prediction checks require the private review package. Independently repeating inference requires the data and corresponding weights, or retraining from the recorded recipe.

Datasets, model binaries, passwords, SSH connection helpers and private manuscript drafts are excluded from publication. Original local files and Git history are retained.
