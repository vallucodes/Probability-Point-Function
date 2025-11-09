#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <random>
#include <boost/math/distributions/normal.hpp>

#include "InverseCumulativeNormal.h"

#define SAMPLES_1e7 10000000
#define SAMPLES_1e8 100000000

void	test_monotonicity_whole_range() {
	quant::InverseCumulativeNormal icn;

	const int N = SAMPLES_1e7;
	// std::cout << "Step size = " << 1.0 / N << std::endl;
	double x, z;
	double old_z = 0.0;
	double old_x = 0.0;
	bool fail = false;

	old_x = (0 + 0.5) / N;
	old_z = icn(old_x);
	for (int i = 1; i < N; i++)
	{
		x = (i + 0.5) / N;
		z = icn(x);
		std::cout << std::fixed << std::setprecision(18);
		// std::cout << "Comparing: " << z << " < " << old_z << std::endl;
		// std::cout << "Diff = " << z - old_z << std::endl;
		if (z < old_z)
		{
			std::cout << "Monotonicity error: values:\n"
			<< "i = " << i
			<< "\nx = " << x
			<< "\nz = " << z
			<< "\nold_x = " << old_x
			<< "\nold_z = " << old_z
			<< std::endl;
			fail = true;
		}
		old_z = z;
		old_x = x;
	}
	if (fail == false)
		std::cout << "Monotonicity test for whole range passed ✅\n";
	else
		std::cout << "Monotonicity test for whole range failed ❌\n";
}

void	test_monotonicity_tail_and_join() {
	quant::InverseCumulativeNormal icn;

	const int N = SAMPLES_1e8;
	const double upper_value_test_range = 0.21;
	// std::cout << "Step size: " << 1.0 / N << std::endl;
	double x, z;
	double old_z = 0.0;
	double old_x = 0.0;
	bool fail = false;

	old_x = (0 + 0.5) / N * upper_value_test_range;
	old_z = icn(old_x);
	for (int i = 1; i < N; i++)
	{
		x = (i + 0.5) / N;
		z = icn(x);
		std::cout << std::fixed << std::setprecision(18);
		// std::cout << "Comparing: " << z << " < " << old_z << std::endl;
		// std::cout << "Diff = " << z - old_z << std::endl;
		if (z < old_z)
		{
			std::cout << "Monotonicity error: values:\n"
			<< "i = " << i
			<< "\nx = " << x
			<< "\nz = " << z
			<< "\nold_x = " << old_x
			<< "\nold_z = " << old_z
			<< std::endl;
			fail = true;
		}
		old_z = z;
		old_x = x;
	}
	if (fail == false)
		std::cout << "Monotonicity test for tail and join passed ✅\n";
	else
		std::cout << "Monotonicity test for tail and join failed ❌\n";
}

double	step_ulp(double x, int k, double toward) {
	for (int i = 0; i < k; ++i)
		x = std::nextafter(x, toward);
	return x;
}

void	test_monotonicity_two_random_adjacent_points() {

	quant::InverseCumulativeNormal icn;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	const int N = SAMPLES_1e7;
	bool fail = false;

	for (int i = 1; i < N; i++)
	{
		const double x1 = dist(gen);
		const double x2 = step_ulp(x1, 100, 1.0);

		const double z1 = icn(x1);
		const double z2 = icn(x2);

		if (z2 <= z1)
		{
			// fails++;
			std::cout << std::fixed << std::setprecision(50);
			std::cout << "x1 = " << x1 << "\n";
			std::cout << "x2 = " << x2 << "\n";
			std::cout << "z1 = " << z1 << "❌\n";
			std::cout << "z2 = " << z2 << "❌\n\n";
			fail = true;
		}
	}

	if (fail == false)
		std::cout << "Monotonicity adjacent points test for tail and join passed ✅\n";
	else
		std::cout << "Monotonicity adjacent points test for tail and join failed ❌\n";
}

void test_symmetry() {
	quant::InverseCumulativeNormal icn;
	const int N = SAMPLES_1e7;
	double max_diff = 0.0;

	for (int i = 1; i < N; i++)
	{
		double x = i / N;
		if (x == 0.0)
			continue;

		double z1 = -icn(x);
		double z2 = icn(1.0 - x);

		double diff = std::abs(z1 - z2);
		if (diff > max_diff)
			max_diff = diff;
	}

	if (max_diff < 1e-16)
		std::cout << "Symmetry test passed ✅\n";
	else
		std::cout << "Symmetry test failed ❌\n";
}

static inline double phi(double z) {
	// 1/sqrt(2π) * exp(-z^2 / 2)
	constexpr double INV_SQRT_2PI =
		0.398942280401432677939946059934381868475858631164934657; // 1/sqrt(2π)
	return INV_SQRT_2PI * std::exp(-0.5 * z * z);
}

void	test_derivative() {
	quant::InverseCumulativeNormal icn;
	const int N = SAMPLES_1e7;
	double relative_error = 0.0;

	for (int i = 1; i < N; i++)
	{
		double x = static_cast<double>(i) / N;
		const double delta = 1e-8;

		const double gx_plus = icn(x + delta);
		const double gx_minus = icn(x - delta);

		const double num_derivative = (gx_plus - gx_minus) / (2 * delta);
		const double inv_phi_gx = 1.0 / phi(icn(x));

		relative_error += std::abs((num_derivative - inv_phi_gx) / inv_phi_gx);
	}

	const double mean_relative_error = relative_error / (N - 1);
	if (mean_relative_error < 1e-06)
		std::cout << "Derivative sanity test passed ✅\n";
	else
		std::cout << "Derivative sanity test failed ❌\n";
}

void	test_single_value() {
	quant::InverseCumulativeNormal icn;
	boost::math::normal_distribution<> standard_normal;

	const double x = 0.976314033801166214487921024556271731853485107421875;

	const double z_approx = icn(x);
	const double z_ref = boost::math::quantile(standard_normal, x);
	const double error = std::abs(z_approx - z_ref);

	std::cout << std::scientific << std::setprecision(30);
	std::cout << "error = " << error << "\n";
}

void test_symmetry_single_value() {
	quant::InverseCumulativeNormal icn;
	boost::math::normal_distribution<> standard_normal;

	double x1 = 0.023606390170104123160665920977407949976623058319091796875;
	double x2 = 1.0 - x1;

	const double z_approx1 = icn(x1);
	const double z_approx2 = icn(x2);
	const double diff = std::abs(z_approx1 + z_approx2);

	std::cout << std::fixed << std::setprecision(30);
	std::cout << "diff = " << diff << "\n";
}

void	stats_errors_random_points() {
	std::cout << "\n=== Calculation of error Φ(ẑ) ≈ x over random points ===\n";
	quant::InverseCumulativeNormal icn;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(1e-12, 1.0 - 1e-12);

	boost::math::normal_distribution<> standard_normal;

	const int N = SAMPLES_1e7;

	double error_sum = 0.0;
	double error_max = 0.0;

	std::vector<double> errors;
	errors.reserve(N);

	for (int i = 0; i < N; i++)
	{
		const double x = dist(gen);
		const double z_approx = icn(x);
		const double z_ref = boost::math::quantile(standard_normal, x);
		const double error = std::abs(z_approx - z_ref);
		errors.push_back(error);
		error_sum += error;
		if (error > error_max)
			error_max = error;
	}
	std::sort(errors.begin(), errors.end());
	const double error_avg = error_sum / N;
	const double error_p99 = errors[static_cast<size_t>(0.99 * N)];
	std::cout << std::scientific << std::setprecision(4);
	std::cout << "  (Random Points: " << N << " samples)\n";
	std::cout << "error mean = " << error_avg << "\n";
	std::cout << "error p99 = " << error_p99 << "\n";
	std::cout << "error max = " << error_max << "\n";
}

void stats_errors_linear_spacing() {
	std::cout << "\n=== Calculation of error Φ(ẑ) ≈ x over linear grid ===\n";
	quant::InverseCumulativeNormal icn;
	boost::math::normal_distribution<> standard_normal;

	const int N = SAMPLES_1e7;
	double error_sum = 0.0;
	double error_max = 0.0;

	std::vector<double> errors;
	errors.reserve(N);

	const double x_min = 1.0e-12;
	const double x_max = 1.0 - 1.0e-12;
	const double range = x_max - x_min;

	for (int i = 0; i < N; i++)
	{
		const double x = x_min + (static_cast<double>(i) / N) * range;
		if (x >= x_max)
			break;

		const double z_approx = icn(x);
		const double z_ref = boost::math::quantile(standard_normal, x);
		const double error = std::abs(z_approx - z_ref);

		errors.push_back(error);
		error_sum += error;
		error_max = std::max(error_max, error);
	}

	std::sort(errors.begin(), errors.end());
	const double error_avg = error_sum / N;

	const double error_p99 = errors[static_cast<size_t>(0.99 * N)];

	std::cout << std::scientific << std::setprecision(4);
	std::cout << "  (Linear Points: " << N << " samples)\n";
	std::cout << "error mean = " << error_avg << "\n";
	std::cout << "error p99 = " << error_p99 << "\n";
	std::cout << "error max = " << error_max << "\n";
}

int main() {

	test_monotonicity_whole_range();
	test_monotonicity_tail_and_join();
	test_monotonicity_two_random_adjacent_points();
	test_symmetry();
	test_derivative();
	// test_symmetry_single_value();
	// test_single_value();

	stats_errors_random_points();
	stats_errors_linear_spacing();

	return 0;
}


