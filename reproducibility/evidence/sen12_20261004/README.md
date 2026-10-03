# Sen12 local source-data recovery — 2026-10-04

Three user-downloaded Raw/Sentinel-2 archives match the publisher's SHA256 values at dataset revision `311f426d0e2fa9b532772fbee0641e2f9db6da00`. Only Chimanimani files were retained. Original archives and original NetCDF values remain intact.

| Check | Result |
|---|---:|
| Retained files | 1,133 |
| Retained bytes | 6,997,828,270 |
| Files with positive mask pixels | 423 |
| Files with all-zero masks | 710 |
| Pinned official LD candidates for Chimanimani | 349 |
| Missing official candidates | 0 |
| Official versus observed positive-pixel count disagreements | 0 |
| Structural file errors in final audit | 0 |
| Exact spectral-array duplicate groups | 0 |
| Pairs of patch footprints with positive-area overlap | 0 |

The official regional split contains 223 train, 55 validation and 71 test samples. The earlier planning count of 699 was incorrect: it combined sample IDs, modality paths and the inventory name. The corrected parser counts only the `s2` file field and rejects duplicate filenames across splits. These official samples are not the historical custom 40/10/210 split.

## Quality findings

All 13 data variables were read. Each has shape `(time, x, y) = (15, 128, 128)`; no NaN or Inf was observed. All masks are binary and constant across time. The CRS is EPSG:32736 with 10 m spacing. Projected coordinate values must not be interpreted as longitude and latitude. RGB bands are identified by variable names B04/B03/B02.

Every retained file contains at least one time frame with SCL=255; 3,285 frames contain this code. It is present in the hash-verified source and has no declared SCL `_FillValue` in the inspected files. We retain it as unknown quality, rather than interpreting it as a standard scene class or damaged download. The initial scan flagged the code as a structural error; the final audit separates structural validity from quality policy and records it explicitly. No file was removed because of it.

All 423 positive-mask samples have event date 2019-03-15. Their recorded post-event frames contain no SCL=255; eight are dated on the event day itself. The 710 zero-mask samples have no per-sample event date. A subsequent single-frame experiment must freeze date selection, handling of unknown quality, and the regional event rule before evaluating models.

No geographic-overlap finding establishes independence of neighboring tiles or related events. A temporal sequence from one location must remain in one split. Historical custom IDs, generating scripts and the corresponding Sen12 checkpoint have not been recovered. No new external model score or claim of historical reproduction is made.

## Reproduce the preparation

Obtain the three official files under their dataset access conditions. Keep the JSON archive manifest and official split file at the pinned revisions below. Use a fresh output directory:

```bash
python -m reproducibility.prepare_sen12_subset \
  --archives-dir /path/to/downloads \
  --official-manifest /path/to/official_archives.json \
  --official-splits /path/to/official_ld_raw_s2_splits.json \
  --out /path/to/chimanimani_prepared

python -m reproducibility.audit_sen12_subset \
  --prepared /path/to/chimanimani_prepared \
  --out /path/to/chimanimani_prepared/final_audit

python -m unittest discover -s tests -p test_sen12_preparation.py -v
```

The recovery has six passing tests covering path identity, duplicate split entries, archive paths, CF calendar handling, unknown SCL quality and footprint intersections. Raw data and transfer archives are stored separately; Git contains code and audit metadata only.

## Sources

- [Sen12Landslides dataset and CC BY 4.0 attribution](https://huggingface.co/datasets/paulhoehn/Sen12Landslides).
- [Pinned Raw/S2 archives](https://huggingface.co/datasets/paulhoehn/Sen12Landslides/tree/311f426d0e2fa9b532772fbee0641e2f9db6da00/data_raw/s2).
- [Pinned official LD/Raw/S2 split](https://github.com/PaulH97/Sen12Landslides/blob/590a726d804e9db6b967d0f34e41bbfd8400a1c0/tasks/S12LS-LD/raw/s2/splits.json).
