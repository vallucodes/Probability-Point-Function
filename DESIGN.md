# Probability-Point-Function

## Fitting

### Center

Degrees m/n = 8/8. 2000 samples. \
Fitting range [0.5, 0.999999]. \
Usage range [0.02, 0.98]. \
Ridge λ = 1e-16

Chebyshev-like nodes meaning fewer nodes toward the center and more towards the join region near 0.98. Weights applied to 30% upper part of range of samples. Starting from weight = 2.0 linearly increasing to 3.0 near 0.98. Solves for θ using least squares on a linear system.

### Tail

Degrees p/q = 8/8. 200 samples. \
Fitting range [1e-16, 0.02]. \
Testing range [1e-12, 0.02] and [0.98, 1 - 1e-12]. \
Ridge λ = 0.1

Distribute samples using log spacing in range [1e-16, 0.002] and using linear spacing in range (0.002, 0.02]. Upper and lower 30% of the samples are weighted linearly in range [2,3]. The lower boundary chosen by observing the start of curvature. Upper boundary represents join area. Solves for θ using least squares on a linear system.

## Error evaluation

| Case | Mean Error | p99 Error | Max Error |
|------|-------------|-----------|-----------|
| **Center [0.02, 0.98] after fitting** | 5.06e-07 | 8.28e-06 | 2.87e-05 |
| **Tail [1e-12, 0.02] after fitting** | 1.78e-05 | 4.07e-05 | 4.44e-05 |
| **Whole [1e−12 , 1 − 1e−12] range after Halley’s steps** | 1.32e-16 | 5.22e-16 | 1.51e-15 |

Halley's residual computed with numerically stable method for tail (x ≤ 1e-8 or x ≥ 1 - 1e-8) and direct residual method used for central part. It uses 0, 1 or 2 steps to achieve sufficient accuracy.

## Performance Measurements

| Case | Notes | Method | Calls | Total Time (ms) | Avg Time (ns/call) | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Non-vectorized** | Baseline Reference | Bisection | $10^7$ | 8620.24 | 862.024 | 1× |
|  | **Algorithmic Gain** | **Rational functions** | ${10^7}$ | 489.763 | 48.9763 | **~17×** |
| **2. Vectorized** | Reference for Parallel | Vector Baseline | $10^6$ | 47.2588 | 47.2589 | 1× |
|  | **Multi-Core Parallelism** | **Vector + OpenMP** | ${10^6}$ | 7.16557 | 7.16557 | **~6×** |

Every timing is using at least `-O3 -ffast-math -march=native` optimization flags.

Non-vectorized are single-argument timed results. Baseline is slow bisection model and second row represents fast core using Horner's method to evaluate. Speedup is due to different method.

Both vectorized result use the faster core method, but the difference in runs is multicore usage. First timing uses single-threaded loop. Second timing uses `#pragma omp parallel for` multicore optimization compiled with additional flag `-fopenmp`. Speedup is due to parallelization.

## Non-idealities and limitations

Floating-point limit at around 1e-16 decimals limits the precision of final value z. Implementation guarantees monotonicity only when adjacent inputs differ by a minimum of 100 ULPs, serving as a stable margin against accumulated rounding error. Valid usage range that is tested and validated is [1e-12, 1 - 1e-12].
