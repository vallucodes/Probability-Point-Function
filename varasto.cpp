#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <algorithm>
#include <iostream>

// #define ICN_ENABLE_HALLEY_REFINEMENT

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

		if (x > 0.5)
		{
			double xm = std::nextafter(1.0 - x, 0.0);
			return -standard_value(xm);
		}

		std::cout << std::fixed << std::setprecision(30);
		std::cout << "x before anything: " << x << std::endl;
		// Piecewise structure left in place so you can drop in rational approximations.
		if (x < x_low_)
		{
			double z = tail_value_baseline(x);
			#ifdef ICN_ENABLE_HALLEY_REFINEMENT
				z = halley_refine(z, x);
				z = halley_refine(z, x);
			#endif   // << replace with tail mapping + rational
			return z;
		}

		else if (x > x_high_)
		{
			double z = -tail_value_baseline(1.0 - x);
			#ifdef ICN_ENABLE_HALLEY_REFINEMENT
				z = halley_refine(z, x);
				z = halley_refine(z, x);
			#endif
			return z;
		}

		else if (x > 0.5)
		{
			double z = central_value_baseline(x); // << replace with central-region rational
			#ifdef ICN_ENABLE_HALLEY_REFINEMENT
				z = halley_refine(z, x);
			#endif
			return z;
		}
		else
		{
			double z = -central_value_baseline(1 - x); // << replace with central-region rational
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

	static inline double Q_stable(double z) {
		constexpr double INV_SQRT_2 =
			0.707106781186547524400844362104849039284835937688474036588; // 1/√2
		return 0.5 * std::erfc(z * INV_SQRT_2);
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
		// std::cout << std::fixed << std::setprecision(30);
		// std::cout << "x: " << x << "\n";
		// std::cout << "u: " << u << "\n";
		// std::cout << "r: " << r << "\n";

		double P = 0.0;
		auto it = P_coeffs.end();
		while (it != P_coeffs.begin()) {
			--it;
			P = *it + P * r;
		}
		// std::cout << "P (Horner result): " << P << "\n";

		double Q = 0.0;
		it = Q_coeffs.end();
		while (it != Q_coeffs.begin()) {
			--it;
			Q = *it + Q * r;
		}
		Q = 1 + Q * r;
		// std::cout << "Q (Horner result): " << Q << "\n";

		double z = u * P / Q;
		// std::cout << "u * P / Q (raw z): " << z << "\n\n";

		return z;
	}

	static inline double tail_value_fast(double x) {
		// Tail region rational approximation (m = 8, n = 8)
		// Valid for x in [10e-16, 0.02] and [0.98, 1 - 10e-16]
		// TODO check which one to use at boundary 0.02/////////////////////////////////////
		// constexpr std::array<double, 9> C_coeffs = {
		// 	-1.4453936962031595e+00,
		// 	5.3437725610366160e-01,
		// 	2.7485580086525579e-01,
		// 	-4.1960419129514329e-02,
		// 	1.0367545320728760e-01,
		// 	-7.0741870362246578e-02,
		// 	-3.2199914177129119e-01,
		// 	3.6027529698610544e-01,
		// 	5.7744261269094987e-02
		// };
		// constexpr std::array<double, 8> D_coeffs = {
		// 	-1.6858265128634287e-01,
		// 	9.5570917623519314e-01,
		// 	-3.8810197773723981e-01,
		// 	4.5056803404836981e-01,
		// 	-8.8167429736128899e-02,
		// 	3.6287243331601549e-01,
		// 	5.7701387291593187e-02,
		// 	4.5659896463767780e-07
		// };

		// Tail region rational approximation (m = 8, n = 8)
		// Valid for x in [1e-15, 0.02]
		// constexpr std::array<double, 9> C_coeffs = {
		// 	-1.5933036468058817e-01,
		// 	-2.6866775840074356e-02,
		// 	1.2561594429529291e-01,
		// 	9.1658473821320016e-02,
		// 	-5.1239703216847204e-02,
		// 	5.3673202669828612e-02,
		// 	-1.0158308557131401e-01,
		// 	4.1317999583923813e-02,
		// 	5.2199325727007653e-02
		// };
		// constexpr std::array<double, 8> D_coeffs = {
		// 	-2.6000740241709419e-01,
		// 	2.7436873543222867e-04,
		// 	2.5551338796076700e-01,
		// 	-7.0922984723620269e-02,
		// 	5.3863707937882303e-02,
		// 	4.6568916564036104e-02,
		// 	5.2052500816583122e-02,
		// 	2.3220097455575739e-06
		// };

		// Tail region rational approximation (m = 8, n = 8)
		// Valid for x in [1e-15, 0.02]
		constexpr std::array<double, 9> C_coeffs = {
			-6.0042308945282707e-01,
			1.0758930509662520e-01,
			3.0332610475203353e-01,
			-3.3804033257979489e-02,
			1.3382873627985800e-01,
			-2.9580458413863892e-02,
			3.3972810405724552e-01,
			-2.1050098774305606e-01,
			-6.7248194629267677e-02
		};
		constexpr std::array<double, 8> D_coeffs = {
			-5.3846100539051089e-01,
			5.1710968386019607e-01,
			1.6853638612244221e-02,
			-1.0463172664394028e-01,
			9.7791265190626628e-02,
			-2.1482641187044155e-01,
			-6.7156570718592754e-02,
			-1.1829534091456484e-06
		};

		double m = std::min(x, 1 - x);
		double t = std::sqrt(-2 * std::log(m));
		double s = std::copysign(1.0, x - 0.5);

		double C = 0.0;
		auto it = C_coeffs.end();
		while (it != C_coeffs.begin())
		{
			--it;
			C = *it + C * t;
		}
		// std::cout << "C: " << C << std::endl;

		double D = 0.0;
		it = D_coeffs.end();
		while (it != D_coeffs.begin())
		{
			--it;
			D = *it + D * t;
		}
		D = 1 + D * t;
		// std::cout << "D: " << D << std::endl;

		return s * C / D;
	}

	// Central-region value
	static inline double central_value_baseline(double x) {
		// No need to check if x > 0.5 or x < 0.5
		// Symmetry is acheieved with the help of u = x - 0.5
		return central_value_fast(x);
	}

	// Tail value
	static inline double tail_value_baseline(double x) {
		// No need to check if x > 0.5 or x < 0.5
		// Symmetry is acheieved with the help of m = std::min(x, 1 - x) and sign s

		return tail_value_fast(x);
	}

#ifdef ICN_ENABLE_HALLEY_REFINEMENT
	// One-step Halley refinement (3rd order). Usually brings result to full double precision.
	static inline double halley_refine(double z, double x) {
		double r;
		// Center region
		if (x > 1e-6 && x < 1 - 1e-6)
		{
			// r = (Φ(z) - x) / φ(z)
			const double f = Phi(z);
			const double p = phi(z);
			r = (f - x) / p;
			// std::cout << std::fixed << std::setprecision(50);

			// std::cout << "center f: " << f << "\n";
			// std::cout << "center x: " << x << "\n";
			// std::cout << "center f - x: " << f - x << "\n";
			// std::cout << "center r: " << r << "\n";

		}
		// Right tail
		else if (x >= 1.0 - 1e-6)
		{
			const double y = 1.0 - x;
			const double phi_val = phi(z);
			const double Q = Q_stable(z);
			const double log_diff = std::log(Q) - std::log(y);
			const double numerator = y * expm1(log_diff);
			r = -numerator / phi_val;
			// std::cout << "phi_val: " << phi_val << "\n";
			// std::cout << "Q: " << Q << "\n";
			// std::cout << "log_diff: " << log_diff << "\n";
			// std::cout << "numerator: " << numerator << "\n";
			// std::cout << "r: " << r << "\n";

			if (r == 0)
			{
				const double f = 1.0 - Q;
				const double x_val = 1.0 - y;
				r = (f - x_val) / phi_val;
				std::cout << "r == 0 case r: " << r << "\n";
			}

		}
		// Left tail
		else
		{
			const double y = x;
			const double phi_val = phi(z);
			const double Q = Phi(z);
			const double log_diff = std::log(Q) - std::log(y);
			const double numerator = y * std::expm1(log_diff);
			r = numerator / phi_val;
		}
		const double denom = 1.0 - 0.5 * z * r;
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


void test_symmetry_delta() {
	quant::InverseCumulativeNormal icn;
	const int N = 100000;
	double max_diff = 0.0;

	double delta_max;
	for (int i = 1; i < N; ++i) {
		double delta = 0.5 * i / N;
		if (delta == 0.0)
			continue;

		double z1 = icn(0.5 + delta);
		double z2 = icn(0.5 - delta);

		double diff = std::abs(z1 + z2);
		if (diff > max_diff)
		{
			max_diff = diff;
			delta_max = delta;
		}
	}

	std::cout << "max |g(1-x) + g(x)| = " << max_diff << "\n";
	std::cout << std::fixed << std::setprecision(100);
	std::cout << "at delta = " << delta_max << "\n";
}
