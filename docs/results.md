# Benchmark results

Reproduce with `balltracker bench --deliveries 150 --ablation-deliveries 60` (arm64 Darwin, 9 workers).

Every delivery is rendered from a seed, so the true 3D path, camera and ball
parameters are known exactly. The camera pose is not taken from the truth: it is
calibrated from pitch landmarks clicked with Gaussian click error, as a user would.

## Main run: 150 deliveries at 240 fps, 1.5 px click error

| Metric | Median | p90 |
|---|---:|---:|
| Stumps-line prediction error, from frames before the pad (cm) | 4.815 | 17.132 |
| Bounce point error (cm) | 5.761 | 17.245 |
| Speed error (km/h) | 1.768 | 14.26 |
| 3D path RMSE over tracked frames (cm) | 16.606 | 54.811 |
| Fit reprojection RMS (px) | 0.493 | 0.849 |
| Tracked position error vs truth (px) | 0.715 | 0.736 |
| Calibration reprojection RMS (px) | 1.777 | 2.209 |
| Restitution error | 0.018 | 0.079 |
| Drag coefficient error (1/m) | 0.001 | 0.002 |

- Deliveries fitted: 148 of 150 (ok: 148, short_track: 2)
- Ball detected in 91.4% of frames where it was visible
- Hitting / missing called correctly: 96.6% (23 of the deliveries would have hit)

### By ball colour

| Ball | Deliveries | Detection recall | Stumps error median (cm) | Hit/miss accuracy |
|---|---:|---:|---:|---:|
| white | 104 | 87.7% | 5.72 | 95.1% |
| red | 46 | 99.7% | 3.37 | 100.0% |

## Ablations

Each row runs the same seeded deliveries with one thing changed.

| Run | Deliveries fitted | Stumps error median (cm) | p90 (cm) | Bounce error median (cm) | Speed error median (km/h) | Hit/miss accuracy |
|---|---:|---:|---:|---:|---:|---:|
| main (240 fps, 1.5 px clicks), first 60 seeds | 59/60 | 5.137 | 18.043 | 5.714 | 1.568 | 96.6% |
| 120 fps | 60/60 | 4.741 | 9.796 | 6.012 | 1.884 | 95.0% |
| 60 fps | 59/60 | 4.838 | 10.973 | 6.072 | 1.66 | 96.6% |
| 30 fps | 54/60 | 6.729 | 12.623 | 6.589 | 1.885 | 94.4% |
| true camera (no click error) | 59/60 | 4.924 | 14.566 | 5.573 | 1.436 | 98.3% |
| 3 px click error | 59/60 | 5.658 | 18.289 | 9.152 | 1.888 | 96.6% |
| no drag in model | 59/60 | 9.198 | 17.422 | 6.698 | 3.825 | 94.9% |
| no swing or seam in model | 59/60 | 8.833 | 34.176 | 7.132 | 9.359 | 89.8% |

## Worst cases in the main run

| Seed | Ball | Status | Stumps error (cm) | Detected / visible frames | Calibration RMS (px) |
|---|---:|---:|---:|---:|---:|
| 1045 | white | short_track | n/a | 42 / 151 | 2.05 |
| 1077 | white | short_track | n/a | 18 / 117 | 1.79 |
| 1080 | white | ok | 64.6 | 110 / 143 | 1.85 |
| 1023 | white | ok | 59.1 | 82 / 163 | 1.51 |
| 1089 | white | ok | 33.6 | 73 / 128 | 1.90 |
| 1052 | white | ok | 30.6 | 102 / 135 | 2.20 |
| 1113 | white | ok | 29.7 | 138 / 163 | 1.98 |
| 1123 | white | ok | 25.0 | 82 / 149 | 1.66 |
