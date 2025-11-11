#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <array>
#include <algorithm>
#include <iostream>

#define ICN_ENABLE_HALLEY_REFINEMENT

namespace quant {

class InverseCumulativeNormal {
public:
	explicit InverseCumulativeNormal(double average = 0.0, double sigma = 1.0)
	: average_(average), sigma_(sigma) {
	}

	// Scalar call: return average + sigma * Φ^{-1}(x)
	inline double operator()(double x) const {
		return average_ + sigma_ * standard_value(x);
	}

	// Vector overload: out[i] = average + sigma * Φ^{-1}(in[i]) for i in [0, n)
	inline void operator()(const double* in, double* out, std::size_t n) const {
		#ifdef ENABLE_OMP
			#pragma omp parallel for
		#endif
		for (std::size_t i = 0; i < n; ++i)
			out[i] = average_ + sigma_ * standard_value(in[i]);
	}

	// Standardized value: inverse CDF with average=0, sigma=1.
	static inline double standard_value(double x) {
		// Handle edge and invalid cases defensively.
		if (x <= 0.0) return -std::numeric_limits<double>::infinity();
		if (x >= 1.0) return  std::numeric_limits<double>::infinity();

		// Use symetry Φ^(-1)(x) = -Φ^(-1)(1 - x)
		if (x > 0.5)
			return -standard_value(1.0 - x);

		// Piecewise structure for rational approximations
		if (x < x_low_) {
			double z = tail_value_baseline(x);
		#ifdef ICN_ENABLE_HALLEY_REFINEMENT
			z = halley_refine(z, x);
		#endif
			return z;
		} else {
			double z = central_value_baseline(x);
		#ifdef ICN_ENABLE_HALLEY_REFINEMENT
			z = halley_refine(z, x);
		#endif
			return z;
		}
	}

private:

	// Standard normal pdf
	static inline double phi(double z) {
		// 1/sqrt(2π) * exp(-z^2 / 2)
		constexpr double INV_SQRT_2PI =
			0.398942280401432677939946059934381868475858631164934657; // 1/sqrt(2π)
		return INV_SQRT_2PI * std::exp(-0.5 * z * z);
	}

	// Standard normal cdf using erfc: Φ(z) = 0.5 * erfc(-z/√2)
	static inline double Phi(double z) {
		constexpr double INV_SQRT_2 =
			0.707106781186547524400844362104849039284835937688474036588; // 1/√2
		return 0.5 * std::erfc(-z * INV_SQRT_2);
	}

	// Approximation for z using rational function in center
	static inline double central_value_fast(double x) {
		// Center region rational approximation (m = 8, n = 8)
		// Valid for x in [0.02, 0.5]
		constexpr std::array<double, 9> P_coeffs = {
			2.5066285790664549e+00,
			-3.0942024238971818e+01,
			1.3082253839727068e+02,
			-1.9191274560397164e+02,
			-2.0028008361062135e+01,
			9.9314827029478010e+01,
			1.0045386960066531e+02,
			1.1306889347740245e+02,
			6.0898710040066099e+01
		};
		constexpr std::array<double, 8> Q_coeffs = {
			-1.3391263734336036e+01,
			6.3910602843373198e+01,
			-1.1890114814438228e+02,
			3.4435593751916244e+01,
			6.7121344042461416e+01,
			5.0944100714769014e+01,
			-2.6189288111109132e+01,
			2.1804001014136286e+01
		};

		double u = x - 0.5;
		double r = u * u;

		double P = 0.0;
		auto it = P_coeffs.end();
		while (it != P_coeffs.begin()) {
			--it;
			P = *it + P * r;
		}

		double Q = 0.0;
		it = Q_coeffs.end();
		while (it != Q_coeffs.begin()) {
			--it;
			Q = *it + Q * r;
		}
		Q = 1 + Q * r;

		double z = u * P / Q;

		return z;
	}

	// Approximation for z using rational function in tails
	static inline double tail_value_fast(double x) {
		// Tail region rational approximation (m = 8, n = 8)
		// Valid for x in [1e-16, 0.02]
		constexpr std::array<double, 9> C_coeffs = {
			2.3423357681893679e-02,
			4.5453227383892833e-02,
			6.8829783426079857e-02,
			6.9198766873887887e-02,
			2.9352393347554642e-02,
			8.6188080834654780e-03,
			2.0270702160544878e-02,
			1.9139428272913046e-02,
			-1.8403871267529012e-02
		};
		constexpr std::array<double, 8> D_coeffs = {
			-2.5742647808853256e-02,
			6.6274907321844942e-03,
			6.8131861396512658e-02,
			5.9588553550430287e-02,
			-1.7093380704801174e-02,
			1.5886452781847131e-02,
			-1.8283158659116609e-02,
			-2.3008144921649043e-06
		};

		double m = std::min(x, 1.0 - x);
		double t = std::sqrt(-2.0 * std::log(m));
		double s = std::copysign(1.0, x - 0.5);

		double C = 0.0;
		auto it = C_coeffs.end();
		while (it != C_coeffs.begin()) {
			--it;
			C = *it + C * t;
		}

		double D = 0.0;
		it = D_coeffs.end();
		while (it != D_coeffs.begin()) {
			--it;
			D = *it + D * t;
		}
		D = 1.0 + D * t;

		double z = s * C / D;
		return z;
	}

	// Central-region value
	static inline double central_value_baseline(double x) {
		return central_value_fast(x);
	}

	// Tail value
	static inline double tail_value_baseline(double x) {
		return tail_value_fast(x);
	}

#ifdef ICN_ENABLE_HALLEY_REFINEMENT
	// One or two -step Halley refinement (3rd order). Usually brings result to full double precision.
	static inline double halley_refine(double z, double x) {
		double r;
		for (int i = 0; i < 2; i++)
		{
			// Center region
			if (x > 1e-8 && x < 1 - 1e-8)
			{
				const double f = Phi(z);
				const double p = phi(z);
				r = (f - x) / std::max(p, std::numeric_limits<double>::min());
			}
			// Tail region
			else
			{
				const double y = x;
				const double phi_val = phi(z);
				const double Q = Phi(z);
				const double log_diff = std::log(Q) - std::log(y);
				const double numerator = y * std::expm1(log_diff);
				r = numerator / std::max(phi_val, std::numeric_limits<double>::min());
			}

			// Checking convergence
			// If residual is below machine precision, skip the Halley step
			if (std::abs(r) < std::numeric_limits<double>::epsilon())
				return z;

			const double denom = 1.0 - 0.5 * z * r;
			z -= r / (denom != 0.0 ? denom
						: std::copysign(std::numeric_limits<double>::infinity(), denom));
		}
		return z;
	}
#endif

	// ---- State & constants ---------------------------------------------------

	double average_, sigma_;

	// Region split (you may adjust in your improved version).
	static constexpr double x_low_  = 0.02;         // ~ Φ(-2.0)
	static constexpr double x_high_ = 1.0 - x_low_;
};

} // namespace quant
