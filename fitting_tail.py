import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from numpy.linalg import lstsq, cond

# samples: Training data samples amount
# n: degree m/n
# x_low, x_high: range for fitting
def fitting(samples = 1000, p = 8, x_low = 10e-16, x_high = 0.02):
	# Distribute samples using log spacing in range [10^-16, 0.002]
	# using linear spacing in range ]0.002, 0.02]
	samples_linear = samples // 3
	samples_log = samples - samples_linear

	start_linear = x_high / 10
	start_log = x_low
	end_log = x_high / 10

	log_start_exp_value = np.log10(start_log)
	log_end_exp_value = np.log10(end_log)

	xs_log = np.logspace(log_start_exp_value, log_end_exp_value, num=samples_log, endpoint=True)
	xs_linear = np.linspace(start_linear, x_high, samples_linear, endpoint=True)
	xs_linear = xs_linear[1:]
	xs = np.concatenate((xs_log, xs_linear))

	# print("xs log:", xs_log)
	# print("xs linear:", xs_linear)
	# print("xs:", xs)

	# Build samples
	A = []
	y = []
	for x in xs:
		z = norm.ppf(x)
		if x - 0.5 < 0:
			s = -1
		else:
			s = 1
		m = min(x, 1 - x)
		t = np.sqrt(-2.0 * np.log(m))

		# Build A and y arrays. y array = z array
		Ai = []
		for j in range(0, p + 1):
			Ai.append(s * (t ** j))

		for j in range(1, p + 1):
			Ai.append(-z * (t ** j))

		A.append(Ai)
		y.append(z)

	A = np.array(A)
	y = np.array(y)



	# Add weights for samples nearing tail
	weights = generate_weights(xs, x_low, x_high)
	# print("weights:", weights)

	A_w = A * weights[:, None]
	y_w = y * weights

	condition_number = np.linalg.cond(A_w)
	print(f"Condition Number of A: {condition_number:.2e}")

	# --- Setup ---
	lambda_val = 1e-10  # Start with lambda = 1e-10 (sqrt(lambda) = 1e-5)
	P = A_w.shape[1] # Number of coefficients (m + n + 1, typically 17)

	# --- 1. Create the Augmentation Matrix ---
	# Augmentation term is sqrt(lambda) * Identity matrix
	lambda_I = np.sqrt(lambda_val) * np.eye(P)

	# --- 2. Augment the Design Matrix (A') ---
	A_augmented = np.vstack([A_w, lambda_I])

	# --- 3. Augment the Target Vector (y') ---
	# Augment y with P zeros
	y_augmented = np.hstack([y_w, np.zeros(P)])

	condition_number = cond(A_augmented)
	print(f"Condition Number (Augumented): {condition_number:.2e}")

	# Solve linear equations with least square method
	theta, residuals, rank, s = np.linalg.lstsq(A_w, y_w, rcond=None)
	# theta, residuals, rank, s = np.linalg.lstsq(A_augmented, y_augmented, rcond=None)

	# print("theta:", theta)
	return theta, x_low, p

# Generate weights for sample points at extreme tail and near-join points
# This could be optimized
def generate_weights(xs, x_low, x_high):

	# Upper 30% is weighted linearly from 2 to 3.
	# Value chosen by observing where function starts to curve
	weighted_range_percentage_extreme_tail = 0.3
	weighted_range_percentage_near_join = 0.3

	# Max and starting values for weightet x's
	weight_start = 2.0
	weight_max = 3.0
	no_weight = 1.0

	weights = []
	sampling_range = x_high - x_low
	x_low_end = x_low + weighted_range_percentage_extreme_tail * sampling_range
	x_high_start = x_high - weighted_range_percentage_near_join * sampling_range

	for x in xs:
		if x <= x_low_end:
			w = weight_start + (weight_max - weight_start) * (x_low_end - x) / (x_low_end - x_low)
			weights.append(w)
		elif x > x_low_end and x < x_high_start:
			weights.append(no_weight)
		elif x >= x_high_start:
			w = weight_start + (weight_max - weight_start) * (x - x_high_start) / (x_high - x_high_start)
			weights.append(w)

	return np.array(weights)

# ----------------------------------------------------------------------------------
# Validation starts here

def	validation(theta, x_low, x_high = 0.02):

	# Calculate x ∈ [10e-16, 0.02]
	xs_validation_left = np.linspace(x_low, x_high, 10000)
	z_real_left = norm.ppf(xs_validation_left)
	z_approx_left = apply_approximation(xs_validation_left, theta)
	error_left = z_real_left - z_approx_left


	# Calculate x ∈ [0.98, 1 - 10e-16]
	xs_validation_right = np.linspace(1 - x_high, 1 - x_low, 10000)
	z_real_right = norm.ppf(xs_validation_right)
	# Symmetry around 0.5 used: Φ^(-1)(x) = -Φ^(-1)(1 - x)
	z_approx_right = -apply_approximation(1 - xs_validation_right, theta)
	error_right = z_real_right - z_approx_right

	# Error stats
	print(f"Max absolute error left: {np.max(np.abs(error_left)):.2e}")
	print(f"Mean absolute error left: {np.mean(np.abs(error_left)):.2e}")

	# print(f"Max absolute error right: {np.max(np.abs(error_right)):.2e}")
	# print(f"Mean absolute error right: {np.mean(np.abs(error_right)):.2e}")
	return xs_validation_left, z_real_left, z_approx_left, error_left, x_high

# Calculate P and Q using Horner's method
def apply_approximation(xs, theta):
	n = (len(theta) - 1) // 2

	s = -1
	m = xs
	t = np.sqrt(-2.0 * np.log(m))

	P_coeffs = theta[:n+1]
	Q_coeffs = theta[n+1:]

	P = 0
	for a in reversed(P_coeffs):
		P = a + P * t

	Q = 0
	for b in reversed(Q_coeffs):
		Q = b + Q * t
	Q = 1 + Q * t

	return s * (P / Q)

def plot(xs_all, z_real_all, z_approx_all, error_all, x_low, x_high):
	fig, axes = plt.subplots(2, 1, figsize=(10, 8))

	# Plot 1: Function comparison
	axes[0].plot(xs_all, z_real_all, 'b-', label='Real Φ⁻¹(x)', linewidth=2)
	axes[0].plot(xs_all, z_approx_all, 'r--', label='Rational approx', linewidth=1.5)
	axes[0].axvline(x_low, color='gray', linestyle=':', alpha=0.5, label='Join points')
	axes[0].axvline(x_high, color='gray', linestyle=':', alpha=0.5)
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

def export(theta, x_low, x_high, p):
	# Export coefficients for C++
	q = p
	print("\n" + "="*60)
	print("C++ COEFFICIENT EXPORT")
	print("="*60)
	print(f"\n// Tail region rational approximation (m = {q}, n = {p})")
	print(f"// Valid for x in [{x_low}, {x_high}]")
	print(f"constexpr std::array<double, {q + 1}> C_coeffs = {{")
	for i, coef in enumerate(theta[:q+1]):
		print(f"    {coef:.16e}{',' if i < q else ''}")
	print("};")
	print(f"constexpr std::array<double, {q}> D_coeffs = {{")
	for i, coef in enumerate(theta[q+1:]):
		print(f"    {coef:.16e}{',' if i < p-1 else ''}")
	print("};")

def main():
	theta, x_low, p = fitting()
	xs_all, z_real_all, z_approx_all, error_all, x_high = validation(theta, x_low)
	# plot(xs_all, z_real_all, z_approx_all, error_all, x_low, x_high)
	export(theta, x_low, x_high, p)

if __name__ == "__main__":
	main()
