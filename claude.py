import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def chebyshev_nodes(n, low, high):
    """Generate Chebyshev nodes mapped to [low, high]"""
    k = np.arange(n)
    t = np.cos((2*k + 1) * np.pi / (2 * n))
    x = 0.5 * (low + high) + 0.5 * (high - low) * t
    return x

# === YOUR APPROACH: Fit from 0.5 to 0.98 ===
x_transition_start = 0.95  # Where you want to start blending to tail
x_high = 0.98              # Upper boundary of central region

m = 6
n = 6
samples = 300

# Generate Chebyshev nodes
xs = chebyshev_nodes(samples, 0.5, x_high)

print(f"Fitting range: [0.5, {x_high}]")
print(f"Transition zone: [{x_transition_start}, {x_high}]")
print(f"Number of samples: {samples}")

# Build linear system
A = []
y = []

for x in xs:
    z = norm.ppf(x)
    u = x - 0.5
    r = u * u

    row = []
    # Numerator: u * (a0 + a1*r + ... + am*r^m)
    for j in range(m + 1):
        row.append(u * (r ** j))
    # Denominator: -(z) * (b1*r + ... + bn*r^n)
    for j in range(1, n + 1):
        row.append(-z * (r ** j))

    A.append(row)
    y.append(z)

A = np.array(A)
y = np.array(y)

# === WEIGHTING STRATEGY ===
# Option 1: Uniform weights (no special treatment)
weights_uniform = np.ones(len(xs))

# Option 2: Slightly upweight the transition zone [0.95, 0.98]
weights_transition = np.ones(len(xs))
for i, x in enumerate(xs):
    if x >= x_transition_start:
        # Gradually increase weight from 1.0 to 2.5
        progress = (x - x_transition_start) / (x_high - x_transition_start)
        weights_transition[i] = 1.0 + 1.5 * progress

# Option 3: Upweight both ends (near 0.5 and near 0.98)
weights_both_ends = np.ones(len(xs))
for i, x in enumerate(xs):
    if x < 0.52:
        # Near 0.5
        weights_both_ends[i] = 2.0
    elif x >= x_transition_start:
        # In transition zone
        progress = (x - x_transition_start) / (x_high - x_transition_start)
        weights_both_ends[i] = 1.0 + 1.5 * progress

# === TRY EACH WEIGHTING SCHEME ===
results = {}

for name, weights in [("uniform", weights_uniform),
                       ("transition", weights_transition),
                       ("both_ends", weights_both_ends)]:

    A_w = A * weights[:, None]
    y_w = y * weights

    # Solve (no ridge for now, as you suggested)
    theta, residuals, rank, s = np.linalg.lstsq(A_w, y_w, rcond=None)

    results[name] = {
        'theta': theta,
        'weights': weights
    }

    print(f"\n{'='*60}")
    print(f"Weighting: {name}")
    print(f"{'='*60}")
    print(f"Weight range: [{weights.min():.2f}, {weights.max():.2f}]")
    print(f"Coefficients a: {theta[:m+1]}")
    print(f"Coefficients b: {theta[m+1:]}")

# === VALIDATION FUNCTION ===
def invnorm_center(x, theta, m, n):
    a = theta[:m+1]
    b = theta[m+1:]

    u = x - 0.5
    r = u * u

    # Horner for P(r)
    P = np.zeros_like(r)
    for coef in reversed(a):
        P = P * r + coef

    # Horner for Q(r) = 1 + b1*r + ... + bn*r^n
    Q = np.zeros_like(r)
    for coef in reversed(b):
        Q = Q * r + coef
    Q = Q * r + 1.0

    return u * (P / Q)

# === COMPARE ALL THREE ===
xs_test = np.linspace(0.5, x_high, 1000)
z_true = norm.ppf(xs_test)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

for idx, (name, result) in enumerate(results.items()):
    theta = result['theta']
    weights = result['weights']

    z_approx = invnorm_center(xs_test, theta, m, n)
    err = z_approx - z_true

    # Function plot
    axes[idx, 0].plot(xs_test, z_true, 'b-', label='True', linewidth=2)
    axes[idx, 0].plot(xs_test, z_approx, 'r--', label='Approx', linewidth=1.5)
    axes[idx, 0].axvline(x_transition_start, color='gray', linestyle=':', alpha=0.5)
    axes[idx, 0].set_title(f'{name}: Function')
    axes[idx, 0].legend()
    axes[idx, 0].grid(True, alpha=0.3)

    # Error plot
    axes[idx, 1].plot(xs_test, err, 'r-', linewidth=1)
    axes[idx, 1].axhline(0, color='k', linewidth=0.8)
    axes[idx, 1].axvline(x_transition_start, color='gray', linestyle=':', alpha=0.5)
    axes[idx, 1].set_title(f'{name}: Error (max={np.max(np.abs(err)):.2e})')
    axes[idx, 1].grid(True, alpha=0.3)
    axes[idx, 1].set_yscale('symlog', linthresh=1e-9)

    print(f"\n{name} errors:")
    print(f"  Max |error|: {np.max(np.abs(err)):.3e}")
    print(f"  Error at x=0.98: {err[-1]:.3e}")
    print(f"  Max error in [0.95, 0.98]: {np.max(np.abs(err[xs_test >= x_transition_start])):.3e}")

plt.tight_layout()
plt.show()

# Show which weighting gives best result in transition zone
print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)
print("Check which weighting scheme gives:")
print("1. Smooth error curve (no spikes)")
print("2. Acceptable max error in [0.95, 0.98]")
print("3. Good overall error distribution")
