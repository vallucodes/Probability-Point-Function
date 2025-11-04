#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <algorithm>
#include <iostream>

namespace quant {

class InverseCumulativeNormal {
public:
	explicit InverseCumulativeNormal(double average = 0.0, double sigma = 1.0)
	: average_(average), sigma_(sigma) {
		std::cout << "Constructor()\n";
	}

	// Scalar call: return average + sigma * Φ^{-1}(x)
	inline double operator()(double x) const {
		std::cout << "Operator(1 arg)\n";
		return average_ + sigma_ * standard_value(x);
	}

	// Vector overload: out[i] = average + sigma * Φ^{-1}(in[i]) for i in [0, n)
	inline void operator()(const double* in, double* out, std::size_t n) const {
		std::cout << "Operator(3 args)\n";
		for (std::size_t i = 0; i < n; ++i) {
			out[i] = average_ + sigma_ * standard_value(in[i]);
		}
	}

	// Standardized value: inverse CDF with average=0, sigma=1.
	// Baseline: deliberately crude but correct bisection. Replace internals with your faster method.
	static inline double standard_value(double x) {
		std::cout << "Standard_value()\n";
		// Handle edge and invalid cases defensively.
		if (x <= 0.0) return -std::numeric_limits<double>::infinity();
		if (x >= 1.0) return  std::numeric_limits<double>::infinity();

		// Piecewise structure left in place so you can drop in rational approximations.
		if (x < x_low_ || x > x_high_) {
			double z = tail_value_baseline(x);   // << replace with tail mapping + rational
		#ifdef ICN_ENABLE_HALLEY_REFINEMENT
			z = halley_refine(z, x);
		#endif
			return z;
		} else {
			double z = central_value_baseline(x); // << replace with central-region rational
		#ifdef ICN_ENABLE_HALLEY_REFINEMENT
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
		// Valid for x in [0.5, 0.98]
		constexpr double center_a[] = {
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

		constexpr double center_b[] = {
			-1.8269566221684713e+01,
			1.3210023191782122e+02,
			-4.6570425537124260e+02,
			7.5421170706875250e+02,
			-2.4339299340854464e+02,
			-5.1237908556421576e+02,
			-6.5600292540741108e+01,
			5.5060171176853385e+02
		};

	}

	static inline double tail_value_fast(double x) {
		// Tail region rational approximation (m = 8, n = 8)
		// Valid for x in [1e-15, 0.02]
		constexpr double tail_a[] = {
			-1.4453936962031595e+00,
			5.3437725610366160e-01,
			2.7485580086525579e-01,
			-4.1960419129514329e-02,
			1.0367545320728760e-01,
			-7.0741870362246578e-02,
			-3.2199914177129119e-01,
			3.6027529698610544e-01,
			5.7744261269094987e-02
		};

		constexpr double tail_b[] = {
			-1.6858265128634287e-01,
			9.5570917623519314e-01,
			-3.8810197773723981e-01,
			4.5056803404836981e-01,
			-8.8167429736128899e-02,
			3.6287243331601549e-01,
			5.7701387291593187e-02,
			4.5659896463767780e-07
		};
	}

	// Baseline central-region value: currently just bisection.
	static inline double central_value_baseline(double x) {
		// std::cout << "central_value_baseline()\n";
		// TODO(candidate): Replace with rational approximation around x≈0.5
		return central_value_fast(x);
	}

	// Baseline tail handler: currently just bisection (slow for extreme x).
	static inline double tail_value_baseline(double x) {
		// std::cout << "tail_value_baseline()\n";
		// TODO(candidate): Implement tail mapping t = sqrt(-2*log(m)) with rational in t
		return tail_value_fast(x);
	}

#ifdef ICN_ENABLE_HALLEY_REFINEMENT
	// One-step Halley refinement (3rd order). Usually brings result to full double precision.
	static inline double halley_refine(double z, double x) {
		// r = (Φ(z) - x) / φ(z)
		const double f = Phi(z);
		const double p = phi(z);
		const double r = (f - x) / std::max(p, std::numeric_limits<double>::min());
		// Halley: z_{new} = z - r / (1 - 0.5*z*r)
		const double denom = 1.0 - 0.5 * z * r;
		return z - r / (denom != 0.0 ? denom
									: std::copysign(std::numeric_limits<double>::infinity(), denom));
	}
#endif

	// ---- State & constants ---------------------------------------------------

	double average_, sigma_;

	// Region split (you may adjust in your improved version).
	static constexpr double x_low_  = 0.02425;         // ~ Φ(-2.0)
	static constexpr double x_high_ = 1.0 - x_low_;
};

} // namespace quant
