/******************************************************************************/
//
//	Copyright (C) 2018, Florian Hirschberger <florian.hirschberger@uol.de>
//
//	LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
//	ACADEMIC FREE LICENSE (AFL) v3.0.
//
/******************************************************************************/

// This source file implements AFK-MC^2 seeding.
// Reference:
// O. Bachem, M. Lucic, H. Hassani, and A. Krause. Fast and provably good
// seedings for k-means. In Proc. Advances in Neural Information Processing
// Systems, pages 55–63, 2016a.

#ifndef ALGORITHM_AFKMC2_HPP
#define ALGORITHM_AFKMC2_HPP

#include <random>
#include "blaze/Blaze.h"

template <typename T>
size_t
afkmc2(
    const blaze::DynamicMatrix<T, blaze::rowMajor>& x,  // dataset or coreset
    const blaze::DynamicVector<double> w,               // weights
          blaze::DynamicMatrix<T, blaze::rowMajor>& s,  // cluster centers
    std::mt19937_64& mt,                                // random number generator
    size_t C,                                           // number of cluster centers
    size_t chain                                        // Markov chain length
) {
    using blaze::sqrNorm;
    using blaze::row;

    size_t N = x.rows();
    size_t D = x.columns();

    size_t count = 0;

    // draw first cluster
    {
        std::discrete_distribution<size_t> i(w.begin(), w.end());
        s.resize(C, D);
        row(s, 0) = row(x, i(mt));
    }

    // compute proposal distribution
    blaze::DynamicVector<double> q(N);
    {
        for (size_t n = 0; n < N; n++) {
            q[n] = sqrNorm(row(x, n) - row(s, 0)) * w[n];
            count++;
        }
        double dsum = 0;
        double wsum = 0;
        for (size_t n = 0; n < N; n++) {
            dsum += q[n];
            wsum += w[n];
        }
        for (size_t n = 0; n < N; n++) {
            q[n] = 0.5 * (q[n] / dsum + w[n] / wsum);
        }
    }

    std::discrete_distribution<size_t> draw_q(q.begin(), q.end());
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    for (size_t c = 1; c < C; c++) {

        // initialize a new Markov chain
        size_t x_idx = draw_q(mt);
        double x_key;

        // compute distance to closest cluster
        double dist = std::numeric_limits<double>::max();
        for (size_t _c = 0; _c < c; _c++) {
            dist = std::min(dist, (double) sqrNorm(row(x, x_idx) - row(s, _c)));
            count++;
        }
        x_key = dist * w[x_idx];

        // Markov chain
        for (size_t i = 1; i < chain; i++) {

            // draw new potential cluster center from proposal distribution
            size_t y_idx = draw_q(mt);
            double y_key;

            // compute distance to closest cluster

            double _dist = std::numeric_limits<double>::max();
            for (size_t _c = 0; _c < c; _c++) {
                _dist = std::min(_dist, (double) sqrNorm(row(x, y_idx) - row(s, _c)));
                count++;
            }
            y_key = _dist * w[y_idx];

            // determine the probability to accept the new sample y_idx
            double y_prob = y_key / q[y_idx];
            double x_prob = x_key / q[x_idx];

            if (x_prob > 0) {
                if ((y_prob / x_prob) > uniform(mt)) {
                    x_idx = y_idx;
                    x_key = y_key;
                }
            } else {
                x_idx = y_idx;
                x_key = y_key;
            }
        }

        row(s, c) = row(x, x_idx);
    }

    return count;
}

#endif
