#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <chrono>
#include <random>
#include <boost/math/distributions/normal.hpp>

#include "InverseCumulativeNormal.h"

void	stats_errors() {
	quant::InverseCumulativeNormal icn;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(1e-12, 1.0 - 1e-12);

	boost::math::normal_distribution<> standard_normal;

	const int N = 10000000;

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
	std::cout << "error avg = " << error_avg << "\n";
	std::cout << "error p99 = " << error_p99 << "\n";
	std::cout << "error max = " << error_max << "\n";
}

void	test_single_value() {

	quant::InverseCumulativeNormal icn;

	double x = 0.976314033801166214487921024556271731853485107421875;

	boost::math::normal_distribution<> standard_normal;

	const double z_approx = icn(x);
	const double z_ref = boost::math::quantile(standard_normal, x);
	const double error = std::abs(z_approx - z_ref);
	std::cout << std::scientific << std::setprecision(4);
	std::cout << "error = " << error << "\n";
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
	std::cout << "Step size = " << 1.0 / N << std::endl;
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
		std::cout << "Diff = " << z - old_z << std::endl;
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
		std::cout << "Diff = " << z - old_z << std::endl;
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
		std::cout << "Monotonicity test for tail and join passed\n";
	else
		std::cout << "Monotonicity test for tail and join failed\n";
}

double step_ulp(double x, int k, double toward = 1.0) {
	for (int i = 0; i < k; ++i)
		x = std::nextafter(x, toward);
	return x;
}

void	test_monotonicity_two_random_adjacent_points() {

	quant::InverseCumulativeNormal icn;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	const int N = 100000000 * 5;
	bool fail = false;

	// int fails = 0;
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
	// std::cout << "fails = " << fails<< "\n";
	if (fail == false)
		std::cout << "Monotonicity test for tail and join passed\n";
	else
		std::cout << "Monotonicity test for tail and join failed\n";
}

void test_symmetry() {
	quant::InverseCumulativeNormal icn;
	const int N = 1000000;
	double max_diff = 0.0;

	double max_x = 0.0;
	for (int i = 1; i < N; i++)
	{
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
	double checksum = 0.0;

	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < N; i++)
		checksum += icn((i + 0.5) / N);
	auto end = std::chrono::high_resolution_clock::now();

	double ms = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << checksum << "\n";
	std::cout << "Elapsed: " << ms << " ms  (avg " << (ms * 1e6 / N) << " ns/call)\n";
}

void	test_speed_vector() {
	quant::InverseCumulativeNormal icn;
	const int N = 1000000;
	const int runs = 10;

	std::vector<double> xin(N);

	for (int i = 0; i < N; i++)
		xin[i] = (i + 1.0) / (N + 1.0);

	std::vector<double> zout(N);
	double total_ms = 0.0;

	for (int i = 0; i < runs; i++)
	{
		auto start = std::chrono::high_resolution_clock::now();
		icn(xin.data(), zout.data(), N);
		auto end = std::chrono::high_resolution_clock::now();
		total_ms += std::chrono::duration<double, std::milli>(end - start).count();
	}

	const double average_ms =  total_ms / runs;

	std::cout	<< "Average elapsed: " << average_ms
				<< " ms  (avg " << (average_ms * 1000000 / (N - 1))
				<< " ns/call)\n";
}

int main() {

	// test_monotonicity_whole_range();
	// test_monotonicity_tail_and_join();
	// test_monotonicity_two_random_adjacent_points();
	// test_speed();
	// test_speed_vector();
	// test_single_value();
	// test_symmetry();
	// test_symmetry_single_value();
	// test_derivative();

	stats_errors();

	return 0;
}


