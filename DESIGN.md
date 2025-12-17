# Probability-Point-Function

## Problem interpretation and approach

Slow bisection method uses many more Φ(z) calls per single evaluation (80 per evaluation) which is very time costly, because Φ(z) is using slow erfc function. Fast Rational method using Horner's method and two optional Halley steps is calling Φ(z) and ϕ(z) each a maximum of 2 times per evaluation making it much faster.

Starting error with fitting is small enough so that only 1 or 2 Halley steps is enough for double precision. Halley's method uses derivative information and is third-order, which allows very fast convergence to desired accuracy.

Vector parallelization speedup is purely based on calculating multiple independent values simultaneuously with multiple cores.

In given instructions there is mistake: this is not true for left tail: Φ(z) = 1 − Q(−z). For left tail this is true: Φ(z) = Q(-z).

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

## Non-idealities and limitations

Floating-point limit at around 1e-16 decimals limits the precision of final value z. Implementation guarantees monotonicity only when adjacent inputs differ by a minimum of 100 ULPs, serving as a stable margin against accumulated rounding error. Valid usage range that is tested and validated is [1e-12, 1 - 1e-12].

## Assumptions, trade-offs, limitations

Floating-point limit at around 1e-16 decimal limits the precision of final value z.

Strict monotonicity is guaranteed up to practical resolution. Testing showed that continuous order reversal occurs when testing inputs separated by 1 ULP due to numerical noise. Implementation guarantees monotonicity only when adjacent inputs differ by a minimum of 100 ULPs, serving as a stable margin against accumulated rounding error.

Valid usage range that is tested and validated is at [1e-12, 1 - 1e-12].

Parallelization optimization performance is dependent on running machines CPU model.

## Testing, numerical stability, performance

### Testing and accuracy

Accuracy target of maximum absolute error of ≤ 1e−4 without refinement is achieved being around 1e-5. After applying 0 to 2 Halley's steps, accuracy is increased to a maximum absolute error of ∼1e−15 (exceeding the 1e-12 goal).

Symmetry, monotonicity and derivative sanity is tested with multiple differently set up tests in tests.cpp achieving the targets. Strict monotonicity is checked with std::nextafter(x) (ULP steps) tests, ensuring the final convergence overcomes calculation noise inherent in double precision limitations.

### Numerical stability

Fitting numerical stability of design matrix is achieved by using Ridge Regularization (λ). For the central fit, λ = 1e-16 was sufficient. For the ill-conditioned tail fit (κ ≈ 1e19), λ = 0.1 was required to drive the final condition number down to ≈1e9.

Refinement in Halley steps uses numerically stable residual calculation formulas (log/expm1 forms) to avoid catastrophic cancellation of the error term (Φ(z)−x) in the tails.

### Performance

Algorithmic replacement of bisection method with Fast Rational Core method with two Optional Halley steps achieved speedup of ~17x.
Multi-core CPU parallelization for vector speedup achieved ~6x speedup, by using OpenMP (#pragma omp parallel for).
