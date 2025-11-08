#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <chrono>
#include <random>
#include <boost/math/distributions/normal.hpp>

#include "InverseCumulativeNormal.h"


// benchmark result bisection
// Elapsed: 13058 ms  (avg 1305.8 ns/call)

// benchmark result fast
// Elapsed: 1235 ms  (avg 123.5 ns/call)


// benchmark result vectors original
// Elapsed: 1549.14 ms  (avg 154.914 ns/call)


void	stats_errors() {
	quant::InverseCumulativeNormal icn; // mean=0, sigma=1

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(1e-12, 1.0 - 1e-12);
	// std::uniform_real_distribution<double> dist(1e-12, 0.5);

	boost::math::normal_distribution<> standard_normal;

	const int N = 10000000;
	double error_sum = 0.0;
	double error_max = 0.0;
	double error_max_x = 0.0;
	for (int i = 1; i < N; i++)
	{
		const double x = dist(gen);
		const double z_approx = icn(x);
		const double z_ref = boost::math::quantile(standard_normal, x);
		const double error = std::abs(z_approx - z_ref);
		error_sum += error;
		if (error > error_max)
		{
			error_max = error;
			error_max_x = x;
		}
	}
	double error_avg = error_sum / (N - 1);
	std::cout << std::scientific << std::setprecision(4);
	std::cout << "error avg: " << error_avg << "\n";
	std::cout << "error max: " << error_max << "\n";
	std::cout << std::fixed << std::setprecision(100);
	std::cout << "at: " << error_max_x << "\n";
}

void	test_single_value() {

	quant::InverseCumulativeNormal icn;

	// double x = 0.9999999246163369104323237479547969996929168701171875;
	// double x = 0.000010000000000000000818030539140313095458623138256371021270751953125;
	double x = 0.976314033801166214487921024556271731853485107421875;

	boost::math::normal_distribution<> standard_normal;

	const double z_approx = icn(x);
	const double z_ref = boost::math::quantile(standard_normal, x);
	const double error = std::abs(z_approx - z_ref);
	// std::cout << std::fixed << std::setprecision(6);
	std::cout << std::scientific << std::setprecision(4);
	std::cout << "error: " << error << "\n";
}

void test_symmetry_single_value() {
	quant::InverseCumulativeNormal icn;
	std::cout << std::fixed << std::setprecision(30);

	double x1 = 0.023606390170104123160665920977407949976623058319091796875;
	double x2 = 1.0 - x1;

	boost::math::normal_distribution<> standard_normal;

	const double z_approx1 = icn(x1);
	const double z_approx2 = icn(x2);

	double diff = std::abs(z_approx1 + z_approx2);
	std::cout << "diff = " << diff << "\n";
}

void	test_monotonicity_whole_range() {
	quant::InverseCumulativeNormal icn;

	const int N = 10000000;
	std::cout << "Step size: " << 1.0 / N << std::endl;
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
		std::cout << "Comparing: " << z << " < " << old_z << std::endl;
		std::cout << "Diff: " << z - old_z << std::endl;
		if (z < old_z)
		{
			std::cout << "Monotonicity error: values:\n"
			<< "i = " << i
			<< "\nx = " << x
			<< "\nz = " << z
			<< "\nold_x: " << old_x
			<< "\nold_z: " << old_z
			<< std::endl;
			fail = true;
		}
		old_z = z;
		old_x = x;
	}
	if (fail == false)
		std::cout << "Monotonicity test for whole range passed\n";
	else
		std::cout << "Monotonicity test for whole range failed\n";
}

void	test_monotonicity_tail_and_join() {
	quant::InverseCumulativeNormal icn;

	const int N = 100000000;
	const double upper_value_test_range = 0.21;
	std::cout << "Step size: " << 1.0 / N << std::endl;
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
		std::cout << "Comparing: " << z << " < " << old_z << std::endl;
		std::cout << "Diff: " << z - old_z << std::endl;
		if (z < old_z)
		{
			std::cout << "Monotonicity error: values:\n"
			<< "i = " << i
			<< "\nx = " << x
			<< "\nz = " << z
			<< "\nold_x: " << old_x
			<< "\nold_z: " << old_z
			<< std::endl;
			fail = true;
		}
		old_z = z;
		old_x = x;
	}
	if (fail == false)
		std::cout << "Monotonicity test for tail and join passed\n";
	else
		std::cout << "Monotonicity test for tail and join failed\n";
}

double step_ulp(double x, int k, double toward = 1.0) {
	for (int i = 0; i < k; ++i)
		x = std::nextafter(x, toward);
	return x;
}

void	test_monotonicity_two_near_random_points() {

	quant::InverseCumulativeNormal icn;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	const int N = 100000000;
	bool fail = false;

	int fails = 0;
	for (int i = 1; i < N; i++)
	{
		const double x1 = dist(gen);
		std::cout << std::fixed << std::setprecision(20);
		// std::cout << "x1: " << x1 << "\n";
		const double x2 = step_ulp(x1, 30, 1.0);
		// std::cout << "x2: " << x2 << "\n";

		const double z1 = icn(x1);
		const double z2 = icn(x2);
		// std::cout << "z1: " << z1 << "\n";
		// std::cout << "z2: " << z2 << "\n\n";

		if (z2 < z1)
		{
			fails++;
			// std::cout << "x1: " << x1 << "\n";
			// std::cout << "x2: " << x2 << "\n";
			// std::cout << "z1: " << z1 << "❌\n";
			// std::cout << "z2: " << z2 << "❌\n\n";
			fail = true;
		}
	}
	std::cout << "fails = " << fails<< "\n";
	if (fail == false)
		std::cout << "Monotonicity test for tail and join passed\n";
	else
		std::cout << "Monotonicity test for tail and join failed\n";
}

void test_symmetry() {
	quant::InverseCumulativeNormal icn;
	const int N = 1000000;
	double max_diff = 0.0;

	double max_x;
	for (int i = 1; i < N; i++) {
		double x = i / N;
		if (x == 0.0)
			continue;

		double z1 = icn(x);
		double z2 = icn(1.0 - x);

		double diff = std::abs(z1 + z2);
		if (diff > max_diff)
		{
			max_diff = diff;
			max_x = x;
		}
	}

	std::cout << "max |g(1-x) + g(x)| = " << max_diff << "\n";
	std::cout << std::fixed << std::setprecision(100);
	std::cout << "at x = " << max_x << "\n";
}

static inline double phi(double z) {
	// std::cout << "phi()\n";
	// 1/sqrt(2π) * exp(-z^2 / 2)
	constexpr double INV_SQRT_2PI =
		0.398942280401432677939946059934381868475858631164934657; // 1/sqrt(2π)
	return INV_SQRT_2PI * std::exp(-0.5 * z * z);
}

void	test_derivative() {
	quant::InverseCumulativeNormal icn;
	const int N = 1000000;
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
	std::cout << std::scientific << std::setprecision(4);
	std::cout << "Average Relative error = " << relative_error / (N - 1) << "\n";
}

void	test_speed() {
	quant::InverseCumulativeNormal icn;
	const int N = 10000000;
	double sum = 0.0;

	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < N; i++)
		sum += icn((i + 0.5) / N);
	auto end = std::chrono::high_resolution_clock::now();

	double ms = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << "Elapsed: " << ms << " ms  (avg " << (ms * 1e6 / N) << " ns/call)\n";
}

void	test_speed_vector() {
	quant::InverseCumulativeNormal icn;
	const int N = 10000000;

	std::vector<double> xin(N);

	for (int i = 0; i < N; i++)
	{
		// std::cout << i << "\n";
		// std::cout << (i + 1.0) / (N + 1.0) << "\n";
		xin[i] = (i + 1.0) / (N + 1.0);
	}

	std::vector<double> zout(N);

	auto start = std::chrono::high_resolution_clock::now();
	icn(xin.data(), zout.data(), N);

	auto end = std::chrono::high_resolution_clock::now();
	double ms = std::chrono::duration<double, std::milli>(end - start).count();

	std::cout << "Elapsed: " << ms << " ms  (avg " << (ms * 1e6 / (N-1)) << " ns/call)\n";
}

int main() {

	// // --- Vector/array usage (multiple values at once) ---

	// test_monotonicity_whole_range();
	// test_monotonicity_tail_and_join();
	// test_monotonicity_two_near_random_points();
	// test_speed();
	test_speed_vector();
	// test_single_value();
	// test_symmetry();
	// test_symmetry_single_value();
	// test_derivative();

	// stats_errors();

	return 0;
}


