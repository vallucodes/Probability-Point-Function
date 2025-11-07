#include <iostream>
#include <iomanip>
#include <array>
#include <chrono>
#include <random>
#include <boost/math/distributions/normal.hpp>

#include "InverseCumulativeNormal.h"


// benchmark result bisection
// Elapsed: 13058 ms  (avg 1305.8 ns/call)

// benchmark result fast
// Elapsed: 1235 ms  (avg 123.5 ns/call)

// int main() {

// }

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

// void	test_symmetry_single_value() {

// 	quant::InverseCumulativeNormal icn;
// 	std::cout << std::fixed << std::setprecision(30);
// 	double delta = 0.499995000000000022755131112717208452522754669189453125;
// 	double x1 = 0.5 + delta;
// 	double x2 = 0.5 - delta;
// 	boost::math::normal_distribution<> standard_normal;

// 	// const double error = std::abs(x2 + x1);

// 	double diff_neg = std::abs(0.5 - x2);
// 	double diff_pos = std::abs(x1 - 0.5);

// 	std::cout << "x1: " << x1 << "\n";
// 	std::cout << "x2: " << x2 << "\n";

// 	std::cout << "Diff Neg vs Delta: " << std::abs(diff_neg - delta) << "\n";
// 	std::cout << "Diff Pos vs Delta: " << std::abs(diff_pos - delta) << "\n";

// 	// std::cout << "error before anything: " << error << std::endl;
// 	const double z_approx1 = icn(x1);
// 	const double z_approx2 = icn(x2);
// 	// const double z_ref = boost::math::quantile(standard_normal, x);
// 	// const double error = std::abs(z_approx - z_ref);
// 	std::cout << std::fixed << std::setprecision(50);
// 	// std::cout << "error: " << error << "\n";
// 	std::cout << "z_approx1: " << z_approx1 << "\n";
// 	std::cout << "z_approx2: " << z_approx2 << "\n";

// 	double diff = std::abs(z_approx1 + z_approx2);
// 	std::cout << "diff = " << diff << "\n";
// }

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
	const double upper_value_test_range = 0.22;
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

void	test_monotonicity_two_near_random_points() {

	quant::InverseCumulativeNormal icn;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	const int N = 100;
	bool fail = false;

	for (int i = 1; i < N; i++)
	{
		const double x1 = dist(gen);
		std::cout << std::fixed << std::setprecision(20);
		std::cout << "x1: " << x1 << "\n";
		const double x2 = std::nextafter(x1, 1.0);
		std::cout << "x2: " << x2 << "\n";

		const double z1 = icn(x1);
		const double z2 = icn(x2);
		std::cout << "z1: " << z1 << "\n";
		std::cout << "z2: " << z2 << "\n\n";

		if (z2 < z1)
		{
			std::cout << "z1: " << z1 << "❌\n";
			std::cout << "z2: " << z2 << "❌\n\n";
			fail = true;
		}
	}
	if (fail == false)
		std::cout << "Monotonicity test for tail and join passed\n";
	else
		std::cout << "Monotonicity test for tail and join failed\n";
}

void test_symmetry() {
	quant::InverseCumulativeNormal icn;
	const int N = 1000000;
	double max_diff = 0.0;

	double delta_max;
	double max_x;
	for (int i = 1; i < N; ++i) {
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

// void test_symmetry_delta() {
// 	quant::InverseCumulativeNormal icn;
// 	const int N = 100000;
// 	double max_diff = 0.0;

// 	double delta_max;
// 	for (int i = 1; i < N; ++i) {
// 		double delta = 0.5 * i / N;
// 		if (delta == 0.0)
// 			continue;

// 		double z1 = icn(0.5 + delta);
// 		double z2 = icn(0.5 - delta);

// 		double diff = std::abs(z1 + z2);
// 		if (diff > max_diff)
// 		{
// 			max_diff = diff;
// 			delta_max = delta;
// 		}
// 	}

// 	std::cout << "max |g(1-x) + g(x)| = " << max_diff << "\n";
// 	std::cout << std::fixed << std::setprecision(100);
// 	std::cout << "at delta = " << delta_max << "\n";
// }

void	test_derivative() {

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

int main() {

	// // --- Vector/array usage (multiple values at once) ---
	// quant::InverseCumulativeNormal icn;
	// const double xin[] = {0.0001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999};
	// double zout[std::size(xin)];
	// icn(xin, zout, std::size(xin)); // out[i] = Φ^{-1}(xin[i])

	// for (std::size_t i = 0; i < std::size(xin); ++i) {
	// 	std::cout << "vector  x=" << xin[i] << "  z=" << zout[i] << "\n";
	// }
	// test_monotonicity_whole_range();
	// test_monotonicity_tail_and_join();
	// test_monotonicity_two_near_random_points();
	// test_speed();
	// test_single_value();
	// test_symmetry();
	// test_symmetry_single_value();
	stats_errors();

	return 0;
}


