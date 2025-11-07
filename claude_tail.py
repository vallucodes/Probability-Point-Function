import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def fit_tail_clean(samples=5000, p=8, q=8, x_high=0.02425):
    """
    Fit RIGHT tail directly: x in [1 - x_high, 1)
    Then use symmetry for left tail
    """
    # We'll fit for the RIGHT tail region [1-x_high, 1.0)
    # Which corresponds to m = 1-x in [0, x_high)

    # Generate samples for m (not x!)
    m_low = 1e-15
    m_high = x_high

    samples_extreme = samples // 3
    samples_mid = samples // 3
    samples_join = samples - samples_extreme - samples_mid

    ms_extreme = np.logspace(np.log10(m_low), -10, samples_extreme, endpoint=False)
    ms_mid = np.logspace(-10, -5, samples_mid, endpoint=False)
    ms_join = np.logspace(-5, np.log10(m_high), samples_join, endpoint=True)

    ms = np.concatenate([ms_extreme, ms_mid, ms_join])

    # Convert to x values (right tail: x = 1 - m)
    xs = 1.0 - ms

    print(f"Fitting RIGHT tail:")
    print(f"  x range: [{xs.min():.10f}, {xs.max():.10f}]")
    print(f"  m range: [{ms.min():.2e}, {ms.max():.6f}]")
    print(f"  Samples: {len(xs)} (p={p}, q={q})")

    # Build system
    A = []
    y = []

    for i, (x, m) in enumerate(zip(xs, ms)):
        z = norm.ppf(x)

        # For RIGHT tail: s = +1
        s = 1.0
        t = np.sqrt(-2.0 * np.log(m))

        # g(x) = s * (c0 + c1*t + ... + cp*t^p) / (1 + d1*t + ... + dq*t^q)
        # Rearrange: s*(c0 + c1*t + ...) - z*(d1*t + ... ) = z

        row = []
        # s * t^j terms (j=0..p)
        for j in range(p + 1):
            row.append(s * (t ** j))
        # -z * t^j terms (j=1..q)
        for j in range(1, q + 1):
            row.append(-z * (t ** j))

        A.append(row)
        y.append(z)

    A = np.array(A)
    y = np.array(y)

    # Weights
    weights = np.ones(len(ms))
    for i, m in enumerate(ms):
        if m < 1e-12:
            weights[i] = 3.0
        elif m < 1e-8:
            weights[i] = 2.0
        elif m > m_high / 2:
            progress = (m - m_high/2) / (m_high - m_high/2)
            weights[i] = 1.0 + 2.0 * progress

    print(f"  Weight range: [{weights.min():.2f}, {weights.max():.2f}]")

    A_w = A * weights[:, None]
    y_w = y * weights

    # Solve with tiny ridge
    lambda_ridge = 1e-14
    ATA = A_w.T @ A_w + lambda_ridge * np.eye(A_w.shape[1])
    ATy = A_w.T @ y_w
    theta = np.linalg.solve(ATA, ATy)

    C = theta[:p+1]
    D = theta[p+1:]

    print(f"\nCoefficients:")
    print("C:", C)
    print("D:", D)

    return C, D, p, q

def eval_tail_right(x, C, D):
    """
    Evaluate RIGHT tail: g(x) = C(t) / D(t) where t = sqrt(-2 log(1-x))
    """
    m = 1.0 - x
    t = np.sqrt(-2.0 * np.log(m))

    # Horner for C(t)
    C_val = 0.0
    for coef in reversed(C):
        C_val = C_val * t + coef

    # Horner for D(t) = 1 + d1*t + d2*t^2 + ...
    D_val = 0.0
    for coef in reversed(D):
        D_val = D_val * t + coef
    D_val = D_val * t + 1.0

    return C_val / D_val

# === FIT ===
print("="*60)
print("CLEAN RIGHT TAIL FIT")
print("="*60)
C, D, p, q = fit_tail_clean(samples=5000, p=8, q=8, x_high=0.02425)

# === VALIDATE ===
print("\n" + "="*60)
print("VALIDATION")
print("="*60)

# Test points
test_xs = [0.98, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
print(f"\n{'x':<15} {'True z':<12} {'Approx z':<12} {'Error':<12}")
for x in test_xs:
    z_true = norm.ppf(x)
    z_approx = eval_tail_right(x, C, D)
    err = z_approx - z_true
    print(f"{x:<15.10f} {z_true:<12.6f} {z_approx:<12.6f} {err:<12.3e}")

# Full range test
xs_test = 1.0 - np.logspace(-15, np.log10(0.02425), 3000)
z_true = norm.ppf(xs_test)
z_approx = np.array([eval_tail_right(x, C, D) for x in xs_test])
err = z_approx - z_true

print(f"\nFull range [{xs_test.min():.10f}, {xs_test.max():.6f}]:")
print(f"  Max |error|: {np.max(np.abs(err)):.3e}")
print(f"  Mean |error|: {np.mean(np.abs(err)):.3e}")
print(f"  RMS error: {np.sqrt(np.mean(err**2)):.3e}")

worst_idx = np.argmax(np.abs(err))
print(f"  Worst at x = {xs_test[worst_idx]:.15f}: {err[worst_idx]:.3e}")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(xs_test, z_true, 'b-', label='True', linewidth=2)
axes[0].plot(xs_test, z_approx, 'r--', label='Approx', linewidth=1.5)
axes[0].set_xlabel('x')
axes[0].set_ylabel('Φ⁻¹(x)')
axes[0].set_title('Right Tail Approximation')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(xs_test, np.abs(err), 'r-', linewidth=1)
axes[1].set_xlabel('x')
axes[1].set_ylabel('|Error|')
axes[1].set_yscale('log')
axes[1].set_title(f'Absolute Error (max: {np.max(np.abs(err)):.3e})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Export
print("\n" + "="*60)
print("C++ EXPORT")
print("="*60)
print(f"constexpr std::array<double, {len(C)}> tail_C = {{")
for i, c in enumerate(C):
    print(f"    {c:.16e}{',' if i < len(C)-1 else ''}")
print("};")
print(f"\nconstexpr std::array<double, {len(D)}> tail_D = {{")
for i, d in enumerate(D):
    print(f"    {d:.16e}{',' if i < len(D)-1 else ''}")
print("};")
