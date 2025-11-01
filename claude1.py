import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def chebyshev_nodes(n, low, high):
    """Generate Chebyshev nodes mapped to [low, high]"""
    k = np.arange(n)
    # Map from [-1, 1] to [low, high]
    t = np.cos((2*k + 1) * np.pi / (2 * n))
    x = 0.5 * (low + high) + 0.5 * (high - low) * t
    return x

def generate_weights(xs, x_low, x_high):
    """
    Generate weights that emphasize the join regions.
    Upweight samples near x_low and x_high where we'll transition to tail approx.
    """
    weights = np.ones_like(xs)

    # Upweight region near the boundaries (transition zones)
    # Use smooth weighting to avoid discontinuities
    transition_width = 0.05  # 5% of range on each side

    for i, x in enumerate(xs):
        # Distance from boundaries
        dist_from_low = abs(x - x_low)
        dist_from_high = abs(x - x_high)

        # Upweight if close to either boundary
        if dist_from_low < transition_width:
            # Smoothly increase weight as we approach x_low
            weights[i] = 1.0 + 2.0 * (1.0 - dist_from_low / transition_width)
        elif dist_from_high < transition_width:
            # Smoothly increase weight as we approach x_high
            weights[i] = 1.0 + 2.0 * (1.0 - dist_from_high / transition_width)

    return weights

# === FIT THE SYMMETRIC CENTRAL REGION ===
# Choose join point (will fit from x_low to x_high)
x_low = 0.000001  # Standard choice
x_high = 1.0 - x_low

# Degree m/n (start with 5/5 or 6/6)
m = 6
n = 6

# Generate samples using Chebyshev nodes
# Only fit the right half [0.5, x_high], exploit symmetry
samples = 500
xs = chebyshev_nodes(samples, 0.5, x_high)

print(f"Fitting range: [{xs.min():.6f}, {xs.max():.6f}]")
print(f"Join point x_high: {x_high:.6f}")

# Build the linear system
A = []
y = []

for x in xs:
    z = norm.ppf(x)
    u = x - 0.5
    r = u * u

    # Row of A: [u*r^0, u*r^1, ..., u*r^m, -z*r^1, ..., -z*r^n]
    row = []
    for j in range(m + 1):
        row.append(u * (r ** j))
    for j in range(1, n + 1):
        row.append(-z * (r ** j))

    A.append(row)
    y.append(z)

A = np.array(A)
y = np.array(y)

# Apply weights
weights = generate_weights(xs, x_low, x_high)
A_weighted = A * weights[:, None]
y_weighted = y * weights

print(f"\nWeight statistics:")
print(f"  Min weight: {weights.min():.2f}")
print(f"  Max weight: {weights.max():.2f}")
print(f"  Mean weight: {weights.mean():.2f}")

# Solve with small ridge regularization to improve conditioning
lambda_ridge = 1e-12
ATA = A_weighted.T @ A_weighted + lambda_ridge * np.eye(A_weighted.shape[1])
ATy = A_weighted.T @ y_weighted
theta = np.linalg.solve(ATA, ATy)

print(f"\nCoefficients (m={m}, n={n}):")
print("a (numerator):", theta[:m+1])
print("b (denominator):", theta[m+1:])

# === VALIDATION ===
def invnorm_center(x, theta, m, n):
    """
    Evaluate rational approximation for x in [x_low, x_high]
    Exploits symmetry: only valid for x >= 0.5
    """
    a = theta[:m+1]
    b = theta[m+1:]

    u = x - 0.5
    r = u * u

    # Horner's method for P(r)
    P = np.zeros_like(r)
    for coef in reversed(a):
        P = P * r + coef

    # Horner's method for Q(r) = 1 + b1*r + ... + bn*r^n
    Q = np.zeros_like(r)
    for coef in reversed(b):
        Q = Q * r + coef
    Q = Q * r + 1.0

    return u * (P / Q)

# Test on dense grid
xs_test = np.linspace(0.5, x_high, 100)
z_true = norm.ppf(xs_test)
z_approx = invnorm_center(xs_test, theta, m, n)
print("z_approx:", z_approx)
err = z_approx - z_true

# Also test symmetry on left half
xs_test_left = np.linspace(x_low, 0.5, 100)
z_true_left = norm.ppf(xs_test_left)
# Use symmetry: Φ^(-1)(0.5 - u) = -Φ^(-1)(0.5 + u)
z_approx_left = -invnorm_center(1.0 - xs_test_left, theta, m, n)
print("z_approx_left:", z_approx_left)
err_left = z_approx_left - z_true_left

# Combine for full range
xs_full = np.concatenate([xs_test_left, xs_test[1:]])  # avoid duplicate at 0.5
z_true_full = np.concatenate([z_true_left, z_true[1:]])
z_approx_full = np.concatenate([z_approx_left, z_approx[1:]])
err_full = np.concatenate([err_left, err[1:]])

print(f"\nError statistics (full central range [{x_low:.6f}, {x_high:.6f}]):")
print(f"  Max |error|: {np.max(np.abs(err_full)):.3e}")
print(f"  Mean |error|: {np.mean(np.abs(err_full)):.3e}")
print(f"  RMS error: {np.sqrt(np.mean(err_full**2)):.3e}")
print(f"  99th percentile |error|: {np.percentile(np.abs(err_full), 99):.3e}")

# Identify worst error location
worst_idx = np.argmax(np.abs(err_full))
print(f"  Worst error at x = {xs_full[worst_idx]:.6f}: {err_full[worst_idx]:.3e}")

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: Function comparison
axes[0].plot(xs_full, z_true_full, 'b-', label='True Φ⁻¹(x)', linewidth=2)
axes[0].plot(xs_full, z_approx_full, 'r--', label='Rational approx', linewidth=1.5)
axes[0].axvline(x_low, color='gray', linestyle=':', alpha=0.5, label='Join points')
axes[0].axvline(x_high, color='gray', linestyle=':', alpha=0.5)
axes[0].set_xlabel('x')
axes[0].set_ylabel('Φ⁻¹(x)')
axes[0].set_title('Central Region Approximation')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Error
axes[1].plot(xs_full, err_full, 'r-', linewidth=1)
axes[1].axhline(0, color='k', linewidth=0.8, linestyle='-')
axes[1].axvline(x_low, color='gray', linestyle=':', alpha=0.5)
axes[1].axvline(x_high, color='gray', linestyle=':', alpha=0.5)
axes[1].set_xlabel('x')
axes[1].set_ylabel('Error')
axes[1].set_title(f'Approximation Error (max: {np.max(np.abs(err_full)):.3e})')
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('symlog', linthresh=1e-8)

plt.tight_layout()
plt.show()

# Export coefficients for C++
print("\n" + "="*60)
print("C++ COEFFICIENT EXPORT")
print("="*60)
print(f"\n// Central region rational approximation (m={m}, n={n})")
print(f"// Valid for x in [{x_low}, {x_high}], use symmetry for [{x_low}, 0.5)")
print("constexpr double center_a[] = {")
for i, coef in enumerate(theta[:m+1]):
    print(f"    {coef:.16e}{',' if i < m else ''}")
print("};")
print("\nconstexpr double center_b[] = {")
for i, coef in enumerate(theta[m+1:]):
    print(f"    {coef:.16e}{',' if i < n-1 else ''}")
print("};")
