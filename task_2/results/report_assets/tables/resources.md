| run | stage | trainable params | connector | LoRA | train time (h) | s/update | peak GPU mem (GiB) |
|---|---|---|---|---|---|---|---|
| vt196 | 1 | 1,837,056 | 1,837,056 | 0 | 0.76 | 1.22 | 1.57 |
| vt196 | 2 | 6,424,576 | 1,837,056 | 4,587,520 | 0.58 | 1.40 | 1.57 |
| vt196 | 3 | 6,424,576 | 1,837,056 | 4,587,520 | 0.28 | 1.36 | 1.57 |
| vt49 | 1 | 1,837,056 | 1,837,056 | 0 | 0.35 | 0.55 | 1.49 |
| vt49 | 2 | 6,424,576 | 1,837,056 | 4,587,520 | 0.34 | 0.82 | 1.49 |
| vt49 | 3 | 6,424,576 | 1,837,056 | 4,587,520 | 0.17 | 0.81 | 1.49 |

Feature caching (separate from training): 9291 images, max shard wall time 1.0 min on 2 GPU(s), 2.61 GiB FP16, peak GPU mem 1.24 GiB.

vt196: dev evaluation total 2.9 min; calibration {'scale': 0.05411031957495429, 'text_median_norm': 0.8745594024658203, 'visual_median_norm_before': 16.162525177001953}; final eval 30.593550711714812 ms/answer

vt49: dev evaluation total 1.6 min; calibration {'scale': 0.07689913291603419, 'text_median_norm': 0.8745594024658203, 'visual_median_norm_before': 11.372812271118164}; final eval 19.8006349843709 ms/answer