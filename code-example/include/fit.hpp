/******************************************************************************/
//
//	Copyright (C) 2018, Florian Hirschberger <florian.hirschberger@uol.de>
//
//	LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
//	ACADEMIC FREE LICENSE (AFL) v3.0.
//
/******************************************************************************/

#include "vc-gmm.hpp"
#include "threads.hpp"
#include "afkmc2.hpp"

template <class F>
std::size_t
converge( const F& f
        , double eps
) {
    std::size_t iterations = 0;

    double previous = 0;
    double current;

    while (true)
    {
        current = f();

        iterations++;

        if (iterations > 1) {
            if (std::abs((current - previous) / current) < eps) {
                break;
            }
        }
        previous = current;
    }

    return iterations;
}

template <typename T>
void
fit( const blaze::DynamicMatrix<T, blaze::rowMajor>& Y
   , const blaze::DynamicVector<double> W
   ,       blaze::DynamicMatrix<T, blaze::rowMajor>& M
   ,       blaze::DynamicVector<double>& P
   , std::size_t seed
   ,       blaze::DynamicMatrix<T, blaze::rowMajor>& S
   , std::size_t K
   , std::size_t G
   , bool R
   , std::size_t chain_length
   , double eps
   , std::size_t nthreads
) {
    tp<true> adaptor(nthreads);
    std::vector<std::mt19937_64> e;
    for (std::size_t t = 0; t < nthreads; t++) {
        std::mt19937_64 mt;
        mt.seed(seed+t);
        mt.discard(1e3);
        e.push_back(mt);
    }
    S = 1.0;

    afkmc2(Y, W, M, e[0], blaze::rows(M), chain_length);

    Dia<T, true> t(Y, W, M, P, K, G, R, e, S, adaptor);

    t.use_fixed_priors = false;
    double value;
    bool needs_variance_guess = true;
    std::size_t shared = 2;
    std::size_t i = 0;

    converge([&]()
    -> double {
        value = std::get<1>(t.E_step(needs_variance_guess));
        needs_variance_guess = false;

        if (i < shared) {
            t.use_fixed_priors = true;
            t.iso_update = true;
        } else {
            t.use_fixed_priors = false;
            t.iso_update = false;
        }

        std::cout << "vcGMM-diag : iteration = " << i << " coreset lower bound = " << value << std::endl; 

        t.M_step();
        i++;

        return value;
    }, eps);

    std::size_t masked = blaze::nonZeros(t.Mask);

    blaze::DynamicMatrix<T, blaze::rowMajor> M_(blaze::rows(M) - masked, blaze::columns(M));
    blaze::DynamicMatrix<T, blaze::rowMajor> S_(blaze::rows(S) - masked, blaze::columns(S));
    blaze::DynamicVector<double>             P_(blaze::rows(S) - masked);

    std::size_t inc = 0;

    for (std::size_t ii = 0; ii < blaze::rows(M); ii++)
    {
        if (t.Mask[ii] == false)
        {
            blaze::row(M_, inc) = blaze::row(M, ii);
            blaze::row(S_, inc) = blaze::row(S, ii);
            P_[inc] = P[ii];
            inc++;
        }
    }
    blaze::swap(M_, M);
    blaze::swap(S_, S);
    blaze::swap(P_, P);
}

template <typename T>
void
fit( const blaze::DynamicMatrix<T, blaze::rowMajor>& Y
   , const blaze::DynamicVector<double> W
   ,       blaze::DynamicMatrix<T, blaze::rowMajor>& M
   ,       blaze::DynamicVector<double>& P
   , std::size_t seed
   ,       blaze::DynamicVector<T>& S
   , std::size_t K
   , std::size_t G
   , bool R
   , std::size_t chain_length
   , double eps
   , std::size_t nthreads
) {
    tp<true> adaptor(nthreads);
    std::vector<std::mt19937_64> e;
    for (std::size_t t = 0; t < nthreads; t++) {
        std::mt19937_64 mt;
        mt.seed(seed+t);
        mt.discard(1e3);
        e.push_back(mt);
    }
    S = 1.0;

    afkmc2(Y, W, M, e[0], blaze::rows(M), chain_length);

    Iso<T, true> t(Y, W, M, P, K, G, R, e, S, adaptor);

    t.use_fixed_priors = false;
    double value;
    bool needs_variance_guess = true;
    std::size_t shared = 2;
    std::size_t i = 0;

    converge([&]()
    -> double {
        value = std::get<1>(t.E_step(needs_variance_guess));
        needs_variance_guess = false;

        if (i < shared) {
            t.use_fixed_priors = true;
            t.iso_update = true;
        } else {
            t.use_fixed_priors = false;
            t.iso_update = false;
        }

        std::cout << "vcGMM-isoflex : iteration = " << i << " coreset lower bound = " << value << std::endl; 

        t.M_step();
        i++;

        return value;
    }, eps);

    std::size_t masked = blaze::nonZeros(t.Mask);

    blaze::DynamicMatrix<T, blaze::rowMajor> M_(blaze::rows(M) - masked, blaze::columns(M));
    blaze::DynamicVector<double> S_(blaze::size(S) - masked);
    blaze::DynamicVector<double> P_(blaze::size(S) - masked);

    std::size_t inc = 0;

    for (std::size_t ii = 0; ii < blaze::rows(M); ii++)
    {
        if (t.Mask[ii] == false)
        {
            blaze::row(M_, inc) = blaze::row(M, ii);
            S_[inc] = S[ii];
            P_[inc] = P[ii];
            inc++;


        }
    }
    blaze::swap(M_, M);
    blaze::swap(S_, S);
    blaze::swap(P_, P);
}

template <typename T>
void
fit( const blaze::DynamicMatrix<T, blaze::rowMajor>& Y
   , const blaze::DynamicVector<double> W
   ,       blaze::DynamicMatrix<T, blaze::rowMajor>& M
   ,       blaze::DynamicVector<double>& P
   , std::size_t seed
   , T& S
   , std::size_t K
   , std::size_t G
   , bool R
   , std::size_t chain_length
   , double eps
   , std::size_t nthreads
) {
    tp<true> adaptor(nthreads);
    std::vector<std::mt19937_64> e;
    for (std::size_t t = 0; t < nthreads; t++) {
        std::mt19937_64 mt;
        mt.seed(seed+t);
        mt.discard(1e3);
        e.push_back(mt);
    }
    S = 1.0;

    afkmc2(Y, W, M, e[0], blaze::rows(M), chain_length);

    Shared<T, true> t(Y, W, M, P, K, G, R, e, S, adaptor);

    t.use_fixed_priors = true;
    double value;
    bool needs_variance_guess = true;
    std::size_t i = 0;

    converge([&]()
    -> double {
        value = std::get<1>(t.E_step(needs_variance_guess));
        needs_variance_guess = false;

        std::cout << "vcGMM-iso : iteration = " << i << " coreset lower bound = " << value << std::endl; 

        t.M_step();
        i++;

        return value;
    }, eps);
}