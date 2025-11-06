#include <iostream>
#include <iomanip>
#include <array>
#include <chrono>
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
	double xs[] = {1e-12, 1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1-1e-6, 1-1e-12};
	// double xs[] = {1 - 1e-12};
	boost::math::normal_distribution<> standard_normal;

	for (double x : xs) {
		double z_approx = icn(x); // z = Φ^{-1}(x)
		double z_ref = boost::math::quantile(standard_normal, x);
		double error = std::abs(z_approx - z_ref);
		// std::cout << "z_approx: " << z_approx << "\n";
		// std::cout << "z_ref: " << z_ref << "\n";
		std::cout << "error: " << error << "\n";
	}
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

// void	test_monotonicity_tail_and_join() {
// 	quant::InverseCumulativeNormal icn;

// 	const int N = 10000;
// 	std::cout << "Step size: " << 1.0 / N << std::endl;
// 	double x, z;
// 	double old_z = 0.0;
// 	double old_x = 0.0;
// 	bool fail = false;

// 	old_x = (0 + 0.5) / N * upper_value_test_range;
// 	old_z = icn(old_x);
// 	for (int i = 1; i < N; i++)
// 	{
// 		x = (i + 0.5) / N;
// 		z = icn(x);
// 		std::cout << std::fixed << std::setprecision(18);
// 		std::cout << "Comparing: " << z << " < " << old_z << std::endl;
// 		std::cout << "Diff: " << z - old_z << std::endl;
// 		if (z < old_z)
// 		{
// 			std::cout << "Monotonicity error: values:\n"
// 			<< "i = " << i
// 			<< "\nx = " << x
// 			<< "\nz = " << z
// 			<< "\nold_x: " << old_x
// 			<< "\nold_z: " << old_z
// 			<< std::endl;
// 			fail = true;
// 		}
// 		old_z = z;
// 		old_x = x;
// 	}
// 	if (fail == false)
// 		std::cout << "Monotonicity test for tail and join passed\n";
// 	else
// 		std::cout << "Monotonicity test for tail and join failed\n";
// }

void	test_symmetry() {

}

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
	stats_errors();

	return 0;
}
