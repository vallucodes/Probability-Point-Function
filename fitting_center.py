import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# samples: Training data samples amount
# n: degree m/n
# x_low, x_high: range used for fitting
# Range used for fitting is wider from final usage range because it seems that
# largest error is at the end of fitting range.
def fitting(samples=200, n=8, x_low=0.5, x_high=0.999999):
	# Distribute samples using Chebyshev logic -> more dense close to tail
	xs = chebyshev_nodes(samples, x_low, x_high)

	# Build samples
	A = []
	y = []
	for x in xs:
		z = norm.ppf(x)
		u = x - 0.5
		r = u * u

		# Build A and y arrays. y array = z array
		Ai = []
		for j in range(0, n + 1):
			Ai.append(u * (r ** j))

		for j in range(1, n + 1):
			Ai.append(-z * (r ** j))

		A.append(Ai)
		y.append(z)

	A = np.array(A)
	y = np.array(y)

	# Add weights for samples nearing tail
	weights = generate_weights(xs, x_low, x_high)
	A_w = A * weights[:, None]
	y_w = y * weights

	# Solve linear equations with least square method
	theta, residuals, rank, s = np.linalg.lstsq(A_w, y_w, rcond=None)
	return theta, x_low, n

# Generate nodes based on Chebyshev grid -> more nodes close to tail for more accuracy
# Optimization: use only upper half for fitting
def chebyshev_nodes(n, x_low, x_high):
	k = np.arange(n)
	t = np.cos((k + 0.5) * np.pi / (2 * n))
	x = x_low + t * (x_high - x_low)
	return x

# Generate weights for sample points near tail region
# This could be optimized
def generate_weights(xs, x_low, x_high):

	# Upper 30% is weighted linearly from 2 to 3.
	# Value chosen by observing where function starts to curve
	weighted_range_percentage = 0.3

	weights = []
	sampling_range = x_high - x_low
	upweight_start = x_high - weighted_range_percentage * sampling_range

	for x in xs:
		if x > upweight_start:
			w = 2.0 + 1.0 * (x - upweight_start) / (x_high - upweight_start)
			weights.append(w)
		else:
			# Weight for the rest is 1
			weights.append(1)

	return np.array(weights)

# ----------------------------------------------------------------------------------
# Validation starts here

# Calculate values for [0.5, 0.98] region, then with help of symmetry get
# values for [0.02, 0.5] region, combine and compare approximation with real PPF
def	validation(theta, x_low, x_high=0.98):
	# Calculate x ∈ [0.5, 0.98]
	xs_validation_right = np.linspace(x_low, x_high, 1000)
	z_real_right = norm.ppf(xs_validation_right)
	z_approx_right = apply_approximation(xs_validation_right, theta, x_low, x_high)
	error_right = z_real_right - z_approx_right

	# Calculate x ∈ [0.02, 0.5]
	xs_validation_left = np.linspace(1 - x_high, x_low, 1000)
	z_real_left = norm.ppf(xs_validation_left)
	# Symmetry around 0.5 used: Φ^(-1)(x) = -Φ^(-1)(1 - x)
	z_approx_left = -apply_approximation(1 - xs_validation_left, theta, x_low, x_high)
	error_left = z_real_left - z_approx_left

	# Combine left and right: x ∈ [0.02, 0.98]
	xs_all = np.concatenate((xs_validation_left, xs_validation_right[1:]))
	z_real_all = np.concatenate((z_real_left, z_real_right[1:]))
	z_approx_all = np.concatenate((z_approx_left, z_approx_right[1:]))
	error_all = np.concatenate((error_left, error_right[1:]))

	# Error stats
	print(f"Max absolute error: {np.max(np.abs(error_all)):.2e}")
	print(f"Mean absolute error: {np.mean(np.abs(error_all)):.2e}")
	return xs_all, z_real_all, z_approx_all, error_all, x_high

# Calculate P and Q using Horner's method
def apply_approximation(xs, theta, x_low, x_high):
	n = (len(theta) - 1) // 2
	u = xs - 0.5
	r = u * u

	P_coeffs = theta[:n+1]
	Q_coeffs = theta[n+1:]

	P = 0
	for a in reversed(P_coeffs):
		P = a + P * r

	Q = 0
	for b in reversed(Q_coeffs):
		Q = b + Q * r
	Q = 1 + Q * r

	return u * (P / Q)

def plot(xs_all, z_real_all, z_approx_all, error_all, x_low, x_high):
	fig, axes = plt.subplots(2, 1, figsize=(10, 8))

	# Plot 1: Function comparison
	axes[0].plot(xs_all, z_real_all, 'b-', label='Real Φ⁻¹(x)', linewidth=2)
	axes[0].plot(xs_all, z_approx_all, 'r--', label='Rational approx', linewidth=1.5)
	axes[0].axvline(x_low, color='gray', linestyle=':', alpha=0.5, label='Join points')
	axes[0].axvline(x_high, color='gray', linestyle=':', alpha=0.5)
	axes[0].axvline(1 - x_high, color='gray', linestyle=':', alpha=0.5)
	axes[0].set_xlabel('x')
	axes[0].set_ylabel('Φ⁻¹(x)')
	axes[0].set_title('Central Region Approximation')
	axes[0].legend()
	axes[0].grid(True, alpha=0.6)

	# Plot 2: Error
	axes[1].plot(xs_all, error_all, 'r-', linewidth=1)
	axes[1].axhline(0, color='k', linewidth=0.8, linestyle='-')
	axes[1].axvline(x_low, color='gray', linestyle=':', alpha=0.5)
	axes[1].axvline(x_high, color='gray', linestyle=':', alpha=0.5)
	axes[1].set_xlabel('x')
	axes[1].set_ylabel('Error')
	axes[1].set_title(f'Approximation Error (max: {np.max(np.abs(error_all)):.3e})')
	axes[1].grid(True, alpha=0.6)
	axes[1].set_yscale('symlog', linthresh=1e-8)

	plt.tight_layout()
	plt.show()

def export(theta, x_low, x_high, n):
	# Export coefficients for C++
	m = n
	print("\n" + "="*60)
	print("C++ COEFFICIENT EXPORT")
	print("="*60)
	print(f"\n// Center region rational approximation (m = {m}, n = {n})")
	print(f"// Valid for x in [{x_low}, {x_high}]")
	print("constexpr double P_coeffs[] = {")
	for i, coef in enumerate(theta[:m+1]):
		print(f"    {coef:.16e}{',' if i < m else ''}")
	print("};")
	print("constexpr double Q_coeffs[] = {")
	for i, coef in enumerate(theta[m+1:]):
		print(f"    {coef:.16e}{',' if i < n-1 else ''}")
	print("};")

def main():
	theta, x_low, n = fitting()
	xs_all, z_real_all, z_approx_all, error_all, x_high = validation(theta, x_low)
	# plot(xs_all, z_real_all, z_approx_all, error_all, x_low, x_high)
	export(theta, x_low, x_high, n)

if __name__ == "__main__":
	main()
