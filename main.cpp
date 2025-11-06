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

int main() {
	quant::InverseCumulativeNormal icn;
	const int N = 10000000;
	double sum = 0.0;

	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < N; i++)
		sum += icn((i + 0.5) / N);
	auto end = std::chrono::high_resolution_clock::now();

	double ms = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << "Elapsed: " << ms << " ms  (avg " << (ms * 1e6 / N) << " ns/call)\n";
	return 0;
}

void	test_monotonicity() {
	quant::InverseCumulativeNormal icn; // mean=0, sigma=1
	// double xs[] = {1e-12, 1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1-1e-6, 1-1e-12};
	double xs[] = {0.99};

	for (double x : xs) {
		double z = icn(x); // z = Φ^{-1}(x)
		std::cout << "scalar  x=" << x << std::setprecision(15) << "\n" << z << "\n";
	}
}

// int main() {
// 	// --- Scalar usage ---
// 	quant::InverseCumulativeNormal icn; // mean=0, sigma=1
// 	double xs[] = {1e-12, 1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1-1e-6, 1-1e-12};
// 	// double xs[] = {1 - 1e-12};
// 	boost::math::normal_distribution<> standard_normal;

// 	for (double x : xs) {
// 		double z_approx = icn(x); // z = Φ^{-1}(x)
// 		double z_ref = boost::math::quantile(standard_normal, x);
// 		double error = std::abs(z_approx - z_ref);
// 		// std::cout << "z_approx: " << z_approx << "\n";
// 		// std::cout << "z_ref: " << z_ref << "\n";
// 		std::cout << "error: " << error << "\n";
// 	}


// 	// // --- Vector/array usage (multiple values at once) ---
// 	// const double xin[] = {0.0001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999};
// 	// double zout[std::size(xin)];
// 	// icn(xin, zout, std::size(xin)); // out[i] = Φ^{-1}(xin[i])

// 	// for (std::size_t i = 0; i < std::size(xin); ++i) {
// 	// 	std::cout << "vector  x=" << xin[i] << "  z=" << zout[i] << "\n";
// 	// }

// 	return 0;
// }
