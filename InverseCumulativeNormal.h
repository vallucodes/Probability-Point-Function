#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <algorithm>
#include <iostream>

#define ICN_ENABLE_HALLEY_REFINEMENT

namespace quant {

class InverseCumulativeNormal {
public:
	explicit InverseCumulativeNormal(double average = 0.0, double sigma = 1.0)
	: average_(average), sigma_(sigma) {
		// std::cout << "Constructor()\n";
	}

	// Scalar call: return average + sigma * Φ^{-1}(x)
	inline double operator()(double x) const {
		// std::cout << "Operator(1 arg)\n";
		return average_ + sigma_ * standard_value(x);
	}

	// Vector overload: out[i] = average + sigma * Φ^{-1}(in[i]) for i in [0, n)
	inline void operator()(const double* in, double* out, std::size_t n) const {
		// std::cout << "Operator(3 args)\n";
		for (std::size_t i = 0; i < n; ++i) {
			out[i] = average_ + sigma_ * standard_value(in[i]);
		}
	}

	// Standardized value: inverse CDF with average=0, sigma=1.
	// Baseline: deliberately crude but correct bisection. Replace internals with your faster method.
	static inline double standard_value(double x) {
		// std::cout << "Standard_value()\n";
		// Handle edge and invalid cases defensively.
		if (x <= 0.0) return -std::numeric_limits<double>::infinity();
		if (x >= 1.0) return  std::numeric_limits<double>::infinity();

		// Use symetry Φ^(-1)(x) = -Φ^(-1)(1 - x)
		if (x > 0.5)
			return -standard_value(1.0 - x);
		// Piecewise structure left in place so you can drop in rational approximations.
		if (x < x_low_ || x > x_high_) {
			double z = tail_value_baseline(x);   // << replace with tail mapping + rational
		#ifdef ICN_ENABLE_HALLEY_REFINEMENT
			// std::cout << "z before first refinement: " << z << "\n";

			z = halley_refine(z, x);
			// std::cout << "z after first refinement: " << z << "\n";

			z = halley_refine(z, x);
			// std::cout << "z after second refinement: " << z << "\n";

		#endif
			return z;
		} else {
			double z = central_value_baseline(x); // << replace with central-region rational
		#ifdef ICN_ENABLE_HALLEY_REFINEMENT
			z = halley_refine(z, x);
			z = halley_refine(z, x);
		#endif
			return z;
		}
	}

private:
	// ---- Baseline numerics (intentionally slow but stable) ------------------

	// Standard normal pdf
	static inline double phi(double z) {
		// std::cout << "phi()\n";
		// 1/sqrt(2π) * exp(-z^2 / 2)
		constexpr double INV_SQRT_2PI =
			0.398942280401432677939946059934381868475858631164934657; // 1/sqrt(2π)
		return INV_SQRT_2PI * std::exp(-0.5 * z * z);
	}

	// Standard normal cdf using erfc: Φ(z) = 0.5 * erfc(-z/√2)
	static inline double Phi(double z) {
		// std::cout << "Phi()\n";
		constexpr double INV_SQRT_2 =
			0.707106781186547524400844362104849039284835937688474036588; // 1/√2
		return 0.5 * std::erfc(-z * INV_SQRT_2);
	}

	// Crude but reliable invert via bisection; brackets wide enough for double tails.
	static inline double invert_bisect(double x) {
		std::cout << "invert_bisect()\n";
		// Monotone Φ(z); find z with Φ(z)=x.
		double lo = -12.0;
		double hi =  12.0;
		// Tighten bracket using symmetry for speed (optional micro-optimization).
		if (x < 0.5) {
			hi = 0.0;
		} else {
			lo = 0.0;
		}

		// Bisection iterations: ~60 is enough for double precision on this interval.
		for (int iter = 0; iter < 80; ++iter) {
			double mid = 0.5 * (lo + hi);
			double cdf = Phi(mid);
			if (cdf < x) {
				lo = mid;
			} else {
				hi = mid;
			}
		}
		return 0.5 * (lo + hi);
	}

	static inline double central_value_fast(double x) {
		// Center region rational approximation (m = 8, n = 8)
		// Valid for x in [0.02, 0.98]
		// TODO check which one to use at boundary 0.02/////////////////////////////////////
		constexpr std::array<double, 9> P_coeffs = {
			2.5066282777485149e+00,
			-4.3170077232190792e+01,
			2.8894237421187995e+02,
			-9.1039065314534901e+02,
			1.1915157857550817e+03,
			4.0463762891090013e+01,
			-8.8453647732874265e+02,
			-6.3172703267590055e+02,
			8.1798896463539131e+02
		};
		constexpr std::array<double, 8> Q_coeffs = {
			-1.8269566221684713e+01,
			1.3210023191782122e+02,
			-4.6570425537124260e+02,
			7.5421170706875250e+02,
			-2.4339299340854464e+02,
			-5.1237908556421576e+02,
			-6.5600292540741108e+01,
			5.5060171176853385e+02
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

	static inline double tail_value_fast(double x) {
		// // Tail region rational approximation (m = 8, n = 8)
		// // Valid for x in [1e-15, 0.02]
		// constexpr std::array<double, 9> C_coeffs = {
		// 	-6.0042308945282707e-01,
		// 	1.0758930509662520e-01,
		// 	3.0332610475203353e-01,
		// 	-3.3804033257979489e-02,
		// 	1.3382873627985800e-01,
		// 	-2.9580458413863892e-02,
		// 	3.3972810405724552e-01,
		// 	-2.1050098774305606e-01,
		// 	-6.7248194629267677e-02
		// };
		// constexpr std::array<double, 8> D_coeffs = {
		// 	-5.3846100539051089e-01,
		// 	5.1710968386019607e-01,
		// 	1.6853638612244221e-02,
		// 	-1.0463172664394028e-01,
		// 	9.7791265190626628e-02,
		// 	-2.1482641187044155e-01,
		// 	-6.7156570718592754e-02,
		// 	-1.1829534091456484e-06
		// };

		// Tail region rational approximation (m = 8, n = 8)
		// Valid for x in [1e-15, 0.02]
		constexpr std::array<double, 9> C_coeffs = {
			-6.0043095946393776e-01,
			1.0759158669250106e-01,
			3.0332939532581726e-01,
			-3.3806155735939918e-02,
			1.3383196789397206e-01,
			-2.9581586513698681e-02,
			3.3973708908486000e-01,
			-2.1050645604756457e-01,
			-6.7250566689214739e-02
		};
		constexpr std::array<double, 8> D_coeffs = {
			-5.3846620747598251e-01,
			5.1711879734577182e-01,
			1.6849725906223252e-02,
			-1.0463289768130692e-01,
			9.7792270337484527e-02,
			-2.1483206574479016e-01,
			-6.7158938217368541e-02,
			-1.1830198695522709e-06
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

		double z = s * (C / D);
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
	// One-step Halley refinement (3rd order). Usually brings result to full double precision.
	static inline double halley_refine(double z, double x) {
		double r;
		// Center region
		if (x > 1e-8 && x < 1 - 1e-8)
		{
			// r = (Φ(z) - x) / φ(z)
			const double f = Phi(z);
			const double p = phi(z);
			// TODO check for division by 0
			r = (f - x) / p;
		}
		// Tail region
		else
		{
			const double y = x;
			const double phi_val = phi(z);
			const double Q = Phi(z);
			const double log_diff = std::log(Q) - std::log(y);
			const double numerator = y * std::expm1(log_diff);
			r = numerator / phi_val;
		}
		const long double denom = 1.0 - 0.5 * z * r;

		return z - r / (denom != 0.0 ? denom
									: std::copysign(std::numeric_limits<double>::infinity(), denom));
	}
#endif

	// ---- State & constants ---------------------------------------------------

	double average_, sigma_;

	// Region split (you may adjust in your improved version).
	static constexpr double x_low_  = 0.02;         // ~ Φ(-2.0)
	static constexpr double x_high_ = 1.0 - x_low_;
};

} // namespace quant
