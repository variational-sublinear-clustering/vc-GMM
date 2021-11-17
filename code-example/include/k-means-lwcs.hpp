/******************************************************************************/
//
//	Copyright (C) 2018, Florian Hirschberger <florian.hirschberger@uol.de>
//
//	LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
//	ACADEMIC FREE LICENSE (AFL) v3.0.
//
/******************************************************************************/

// This source file implements lightweight coreset construction.
// Reference:
// O. Bachem, M. Lucic, and A. Krause. Scalable k-means clustering via
// lightweight coresets. In Proceedings KDD, pages 1119–1127, 2018.

#include <random>

template <typename T>
std::size_t
k_means_lwcs(
	const blaze::DynamicMatrix<T, blaze::rowMajor>& x,
	      blaze::DynamicMatrix<T, blaze::rowMajor>& subsetx,
	      blaze::DynamicVector<double>& subsetw,
    std::mt19937_64& e
) {
	blaze::DynamicVector<double, blaze::rowVector> u(blaze::columns(x), 0.);
	blaze::DynamicVector<double> q(blaze::rows(x));

	for (std::size_t n = 0; n < blaze::rows(x); n++) {
		u += blaze::row(x, n);
	}
	u *= 1.0 / blaze::rows(x);

	for (std::size_t n = 0; n < blaze::rows(x); n++) {
		q[n] = blaze::sqrNorm(blaze::row(x, n) - u);
	}
	double sum = 0;
	for (std::size_t n = 0; n < blaze::rows(x); n++) {
		sum += q[n];
	}
	for (std::size_t n = 0; n < blaze::rows(x); n++) {
		q[n] = 0.5 * (q[n] / sum + 1.0 / blaze::rows(x));
	}

	std::discrete_distribution<size_t> dst(q.begin(), q.end());
	for (std::size_t m = 0; m < blaze::rows(subsetx); m++) {
		std::size_t n = dst(e);
		blaze::row(subsetx, m) = row(x, n);
		subsetw[m] = 1.0 / (q[n] * blaze::rows(subsetx));
	}

	return blaze::rows(x);
}