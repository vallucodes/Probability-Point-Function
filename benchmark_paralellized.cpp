#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

#include "InverseCumulativeNormal.h"

#define RUNS_VECTOR 1000000

void	benchmark_speed_vector() {
	std::cout << "\n=== Vectorized version (parallellized) ===\n";
	quant::InverseCumulativeNormal icn;
	const int N = RUNS_VECTOR;
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

	std::cout	<< "Elapsed: " << average_ms
				<< " ms  (avg " << (average_ms * 1000000 / (N - 1))
				<< " ns/call)\n";
}

int main() {

	// 4. Vectorized version (parallellized)
	benchmark_speed_vector();

	return 0;
}
