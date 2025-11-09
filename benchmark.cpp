#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

#include "InverseCumulativeNormal.h"
#include "InverseCumulativeNormal_baseline.h"

#define RUNS_SINGLE_ARG 10000000
#define RUNS_VECTOR 1000000

void	benchmark_single_arg_slow() {
	std::cout << "\n=== Scalar baseline (slow, bisection) ===\n";
	const int N = RUNS_SINGLE_ARG;
	double checksum = 0.0;

	quant::InverseCumulativeNormal_Baseline icn_slow;

	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < N; i++)
		checksum += icn_slow((i + 0.5) / N);

	auto end = std::chrono::high_resolution_clock::now();

	double ms = std::chrono::duration<double, std::milli>(end - start).count();
	volatile double sink = checksum;
	(void)sink;
	std::cout << "Elapsed: " << ms << " ms  (avg " << (ms * 1e6 / N) << " ns/call)\n";
}

void	benchmark_single_arg_fast() {
	std::cout << "\n=== Scalar optimized (rational approximation) ===\n";
	const int N = RUNS_SINGLE_ARG;
	double checksum = 0.0;

	quant::InverseCumulativeNormal icn_fast;

	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < N; i++)
		checksum += icn_fast((i + 0.5) / N);

	auto end = std::chrono::high_resolution_clock::now();

	double ms = std::chrono::duration<double, std::milli>(end - start).count();
	volatile double sink = checksum;
	(void)sink;
	std::cout << "Elapsed: " << ms << " ms  (avg " << (ms * 1e6 / N) << " ns/call)\n";
}

void	benchmark_speed_vector() {
	std::cout << "\n=== Vectorized version (rational approximation) ===\n";
	const int N = RUNS_VECTOR;
	const int runs = 10;

	quant::InverseCumulativeNormal icn_fast;

	std::vector<double> xin(N);

	for (int i = 0; i < N; i++)
		xin[i] = (i + 1.0) / (N + 1.0);

	std::vector<double> zout(N);
	double total_ms = 0.0;

	for (int i = 0; i < runs; i++)
	{
		auto start = std::chrono::high_resolution_clock::now();
		icn_fast(xin.data(), zout.data(), N);
		auto end = std::chrono::high_resolution_clock::now();
		total_ms += std::chrono::duration<double, std::milli>(end - start).count();
	}

	const double average_ms =  total_ms / runs;

	std::cout	<< "Elapsed: " << average_ms
				<< " ms  (avg " << (average_ms * 1000000 / (N - 1))
				<< " ns/call)\n";
}

int main() {

	// 1. Scalar baseline (slow, bisection)
	benchmark_single_arg_slow();

	// 2. Scalar optimized (rational approximation)
	benchmark_single_arg_fast();

	// 3. Vectorized version (rational approximation)
	benchmark_speed_vector();

	return 0;
}
