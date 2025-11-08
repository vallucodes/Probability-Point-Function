# Probability-Point-Function

## Fitting

### Center

Degrees m/n = 8/8. 200 samples. \
Fitting range [0.5, 0.999999]. \
Usage range [0.02, 0.98].


Chebyshev like nodes meaning less nodes towards center and more towards join-region close to 0.98. Weights applied to 30% upper part of range of samples. Starting from weight = 2.0 linearly increasing to 3.0 closer to 0.98. Solves for the constants (θ) using the least-squares method applied to a linear matrix equation.

### Tail

Degrees p/q = 8/8. 1000 samples. \
Fitting range [10e-16, 0.02]. \
Testing range [10e-12, 0.02] and [0.98, 1 - 10e-12].

Distribute samples using log spacing in range [10^-16, 0.002] and using linear spacing in range ]0.002, 0.02]. Upper and lower 30% are weighted linearly from 2 to 3. Lower boundary chosen by observing curvature starting point. Upper boundary represents join area. Solves for the constants (θ) using the least-squares method applied to a linear matrix equation.

## Error evalaution

### Center after fitting
Max abs error = \
Error p99 = \
Max error =

### Tail after fitting
Max abs error = \
Error p99 = \
Max error =

### After Halleys steps
Max abs error = \
Error p99 = \
Max error =

## Performance measurements

| Case | Method | Calls | Notes | Total Time (ms) | Avg Time (ns/call) | Speedup |
|------|---------|--------|--------|------------------|--------------------|----------|
| **Non-vectorized** | Bisection | 10⁷ | Baseline | 13058 | 1305.8 | 1× |
|  | Rational functions | 10⁷ | Uses rational approximation | 1235 | 123.5 | ~10× |
|  | Rational + optimization flags | 10⁷ | Compiler optimizations enabled | 472.532 | 47.2532 | ~27× |
| **Vectorized** | Vector baseline + Rational functions | 10⁶ | Basic vector implementation | 155.313 | 155.313 | 1× |
|  | Vector + optimization + OpenMP | 10⁶ | Optimized with flags and OpenMP | 7.10344 | 7.10345 | ~21× |


## Non-idealities and limitations

Floating-point limit at around 10^-16 limits the precision of final value z. Monotonicity (not allowing decrease or same result) is achieved with >100 ULP (double) difference between adjacent points.
