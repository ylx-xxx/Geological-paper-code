| Variant | Run | Test_mIoU | Landslide_IoU | Landslide_F1 | Precision | Recall | Input | Channels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RGB-only | input_rgb | 0.691452 | 0.398296 | 0.569688 | 0.610417 | 0.534054 | First 3 channels | 3 |
| 12 MS only | input_ms12 | 0.702396 | 0.420320 | 0.591867 | 0.594164 | 0.589587 | 12 Sentinel-2 bands | 12 |
| RGB + topo | input_rgb_topo | 0.702643 | 0.421530 | 0.593065 | 0.568843 | 0.619442 | First 3 channels + slope + DEM | 5 |
| Full 14-channel | input_full14 | 0.704326 | 0.423090 | 0.594608 | 0.640555 | 0.554811 | 12 MS + slope + DEM | 14 |
