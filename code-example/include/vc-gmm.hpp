/******************************************************************************/
//
//	Copyright (C) 2021, Florian Hirschberger <florian.hirschberger@uol.de>
//
//	LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
//	ACADEMIC FREE LICENSE (AFL) v3.0.
//
/******************************************************************************/

#ifndef REFACTORED_HPP
#define REFACTORED_HPP

//--------------------------------------------------------------------------------------------------

#include <iostream>
#include <cmath>
#include <tuple>

#include "./blaze/Blaze.h"

#include "threads.hpp"

//--------------------------------------------------------------------------------------------------

template <template <typename, bool> class Theta, typename T, bool TPB>
struct Base
{
    const blaze::DynamicMatrix<T, blaze::rowMajor>& Y;  
    const blaze::DynamicVector<double>& W;
          blaze::DynamicMatrix<T, blaze::rowMajor>& M;

    blaze::DynamicVector<double>& P;

    std::size_t N;
    std::size_t D;

    std::size_t C;
    std::size_t C_active;
    std::size_t K;
    std::size_t K_active;
    std::size_t G;
    std::size_t G_active;
    bool        R;

    blaze::DynamicMatrix<std::tuple<std::size_t, double>, blaze::rowMajor> Q;
    blaze::DynamicMatrix<std::tuple<std::size_t, double>, blaze::rowMajor> Search;
    blaze::DynamicMatrix<std::size_t, blaze::rowMajor> GC;
    blaze::DynamicVector<std::size_t> Size;

    blaze::DynamicVector<double> C0;

    blaze::DynamicVector<bool> Mask;

    std::vector<std::mt19937_64>& e;

    tp<TPB>& adaptor;

    double W_sum;
    double tmp_min_sum;

    // ---

    const double threshold = 1.e-300;

    // ---

    bool use_fixed_priors = false;
    bool iso_update = false;

    // ---

    std::size_t                     search(void);
    std::size_t                     compute_all(void);
    double                          distribution(void);
    void                            estimate(void);
    std::tuple<std::size_t, double> E_step(bool guess_variance);
    void                            M_step(void);
    void                            M_step_M(void);
    void                            update(void);

    Base( const blaze::DynamicMatrix<T, blaze::rowMajor>& Y_
        , const blaze::DynamicVector<double>& W_
        ,       blaze::DynamicMatrix<T, blaze::rowMajor>& M_
        ,       blaze::DynamicVector<double>& P_
        , std::size_t K_
        , std::size_t G_
        , bool R_
        , std::vector<std::mt19937_64>& e_
        , tp<TPB>& adaptor_
    )
        : Y       (Y_)
        , W       (W_)
        , M       (M_)
        , P       (P_)
        , N       (blaze::rows   (Y_))
        , D       (blaze::columns(Y_))
        , C       (blaze::rows   (M_))
        , C_active(blaze::rows   (M_))
        , K       (K_)
        , K_active(K_)
        , G       (G_)
        , G_active(G_)
        , R       (R_)
        , Q       (blaze::rows(Y_), K_)
        , C0      (blaze::log (P ))
        , Mask    (blaze::rows(M_), false)
        , e       (e_)
        , adaptor (adaptor_)
        , W_sum   (blaze::sum(W_))
    {
        
        if (K != C) {Search.resize(blaze::rows(Y_), std::min(K_ * G_ + R, blaze::rows(M_)));}
        if (K != C) {GC    .resize(blaze::rows(M_), G_);}
        if (K != C) {Size  .resize(blaze::rows(Y_));}

        if (K == C)
        {
            for (std::size_t n(0); n < N; n++) {
                for (std::size_t k(0); k < K; k++) {
                    std::get<0>(Q(n, k)) = k;
                }
            }
        } else
        {
            blaze::DynamicVector<std::size_t> L(C, std::numeric_limits<std::size_t>::max());

            for (std::size_t n(0); n < N; n++) {
                std::uniform_int_distribution<std::size_t> dst(0, C - 1);
                for (auto& it : blaze::row(Q, n)) {
                    std::size_t u;
                    do {
                        u = dst(e.front());
                    } while (L[u] == n);
                    std::get<0>(it) = u;
                    L[u] = n;
                }
            }
        }

        if (K != C)
        {
            blaze::DynamicVector<std::size_t> L(C, std::numeric_limits<std::size_t>::max());

            for (std::size_t c(0); c < C; c++) {
                std::uniform_int_distribution<std::size_t> dst(0, C - 1);
                GC(c, 0) = c;
                L[c] = c;
                for (auto& it : blaze::subvector(blaze::row(GC, c), 1, G - 1)) {
                    std::size_t u;
                    do {
                        u = dst(e.front());
                    } while (L[u] == c);
                    it = u;
                    L[u] = c;
                }
            }
        }
    }
};


//--------------------------------------------------------------------------------------------------

template <template <typename, bool> class Theta, typename T, bool TPB>
std::size_t
Base<Theta, T, TPB>::compute_all(void)
{
    blaze::DynamicVector<double> tmp_min(
        N, 
        std::numeric_limits<double>::max()
    );

    adaptor.parallel(N, [&] (std::size_t n, std::size_t)
    -> void {
        if (n >= N) {
            throw std::runtime_error("n is weird");
        }
        std::size_t k = 0;
        for (std::size_t j = 0; j < C; j++) {
            if (!Mask[j]) {
                std::get<0>(Q(n, k)) = j;
                if (k >= blaze::columns(Q)) {
                    throw std::runtime_error("k >= blaze::columns(Q)");
                }
                if (n >= blaze::rows(Q)) {
                    throw std::runtime_error("n >= blaze::rows(Q)");
                }
                double tmp;
                std::get<1>(Q(n, k)) = static_cast<const Theta<T, TPB>*>(this)->density(n, std::get<0>(Q(n, k)), tmp);
                tmp_min[n] = std::min((double) tmp_min[n], (double) tmp);
                k++;
            }
        }
    });
    tmp_min_sum = blaze::dot(W, tmp_min);
    return N * K_active;
}

//--------------------------------------------------------------------------------------------------

template <template <typename, bool> class Theta, typename T, bool TPB>
std::size_t
Base<Theta, T, TPB>::search(void)
{
    blaze::DynamicVector<blaze::DynamicVector<std::size_t>> L(
        adaptor.size(), 
        blaze::DynamicVector<std::size_t>(
            C, 
            std::numeric_limits<std::size_t>::max()
        )
    );
    blaze::DynamicVector<double> tmp_min(
        N, 
        std::numeric_limits<double>::max()
    );

    adaptor.parallel(N, [&] (std::size_t n, std::size_t tid)
    -> void {

        Size[n] = 0;
        for (const auto& it : blaze::subvector(blaze::row(Q, n), 0, K_active)) {
            for (const auto& it2 : blaze::subvector(blaze::row(GC, std::get<0>(it)), 0, G_active)) {   
                if (!Mask[it2]) {
                    if (L[tid][it2] != n) {
                        L[tid][it2] = n;
                        std::get<0>(Search(n, Size[n])) = it2;
                        Size[n]++;
                    }
                }
            }
        }

        if (R || (Size[n] < K_active))
        {

            std::uniform_int_distribution<std::size_t> dst(0, C - 1);
            std::size_t it2;
            do {
                do {
                    it2 = dst(e[tid]);
                }
                while(Mask[it2]);
                if (L[tid][it2] != n) {
                    L[tid][it2] = n;
                    std::get<0>(Search(n, Size[n])) = it2;
                    Size[n]++;
                }
            }
            while (Size[n] < K_active);
        }


        for (auto& it : blaze::subvector(blaze::row(Search, n), 0, Size[n])) {
            double tmp;
            std::get<1>(it) = static_cast<const Theta<T, TPB>*>(this)->density(n, std::get<0>(it), tmp);
            tmp_min[n] = std::min((double) tmp_min[n], (double) tmp);
        }
        
        std::nth_element( (blaze::row(Search, n)).begin()
                        , (blaze::row(Search, n)).begin() + K_active
                        , (blaze::row(Search, n)).begin() + Size[n]
                        ,
        [] (auto& lhs, auto& rhs) -> bool {
            return std::get<1>(lhs) > std::get<1>(rhs);
        });

        std::nth_element( (blaze::row(Search, n)).begin()
                        , (blaze::row(Search, n)).begin() + 1
                        , (blaze::row(Search, n)).begin() + K_active
                        ,
        [] (auto& lhs, auto& rhs) -> bool {
            return std::get<1>(lhs) > std::get<1>(rhs);
        });

        blaze::subvector(blaze::row(Q, n), 0, K_active) = blaze::subvector(blaze::row(Search, n), 0, K_active);
    });

    tmp_min_sum = blaze::dot(W, tmp_min);
    return blaze::sum(Size);
}

template <template <typename, bool> class Theta, typename T, bool TPB>
double
Base<Theta, T, TPB>::distribution(void)
{
    blaze::DynamicVector<std::array<double, 64>> FE(
        adaptor.size(), 
        std::array<double, 64>{}
    );

    adaptor.parallel(N, [&] (std::size_t n, std::size_t tid)
    -> void {
        bool found = false;
        double tmp;
        for (const auto& it : blaze::subvector(blaze::row(Q, n), 0, K_active)) {
            if (std::isfinite(std::get<1>(it))) {
                found = true;
                tmp = std::get<1>(it);
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("!found");
        }

        double lim = tmp;
        for (const auto& it : blaze::subvector(blaze::row(Q, n), 0, K_active)) {
            if (std::isfinite(std::get<1>(it))) {
                lim = std::max(lim, std::get<1>(it));
            }
        }
        
        double sum = 0;
        for (auto& it : blaze::subvector(blaze::row(Q, n), 0, K_active)) {
            std::get<1>(it) = blaze::exp(std::get<1>(it) - lim);
            sum += std::get<1>(it);
        }
        for (auto& it : blaze::subvector(blaze::row(Q, n), 0, K_active)) {
            std::get<1>(it) /= sum;
        }
        
        FE[tid][0] += W[n] * (blaze::log(sum) + lim);
    });

    for (std::size_t tid = 1; tid < adaptor.size(); tid++) {
        FE[0][0] += FE[tid][0];
    }
    return FE[0][0];
}

template <template <typename, bool> class Theta, typename T, bool TPB>
void
Base<Theta, T, TPB>::estimate(void)
{
    blaze::DynamicVector<blaze::DynamicVector<std::tuple<std::size_t, std::size_t>>> Tab(
        adaptor.size(), 
        blaze::DynamicVector<std::tuple<std::size_t, std::size_t>>(
            C, 
            std::tuple<std::size_t, std::size_t>{std::numeric_limits<std::size_t>::max(), 0}
        )
    );
    blaze::DynamicVector<blaze::DynamicVector<std::tuple<std::size_t, double, double>>> Sel(
        adaptor.size(), 
        blaze::DynamicVector<std::tuple<std::size_t, double, double>>(
            C, 
            std::tuple<std::size_t, double, double>{std::numeric_limits<std::size_t>::max(), 0, 0}
        )
    );

    auto inverse = std::vector<std::vector<std::size_t>>(
        C,
        std::vector<std::size_t>{}
    );
    for (std::size_t n = 0; n < N; n++) {
        inverse[std::get<0>(Q(n, 0))].push_back(n);
    }
    

    adaptor.parallel(C, [&] (std::size_t c, std::size_t tid)
    -> void {
        if (!Mask[c]) {
            if (inverse[c].size() > 0)
            {
                std::size_t index;
                index = 0;
                for (const auto& n : inverse[c]) {
                    for (const auto& it : blaze::subvector(blaze::row(Search, n), 0, Size[n])) {

                        std::size_t candidate = std::get<0>(it);
                        double density = std::get<1>(it);

                        if (candidate != c)
                        {
                            if (std::get<0>(Tab[tid][candidate]) != c)
                            {
                                std::get<0>(Tab[tid][candidate]) = c;
                                std::get<1>(Tab[tid][candidate]) = index;
                                std::get<0>(Sel[tid][index]) = candidate;
                                std::get<1>(Sel[tid][index]) = 1;
                                std::get<2>(Sel[tid][index]) = density;
                                index++;
                            } else {
                                std::get<1>(Sel[tid][std::get<1>(Tab[tid][candidate])]) += 1;
                                std::get<2>(Sel[tid][std::get<1>(Tab[tid][candidate])]) += density;
                            }
                        }
                    }
                }
                for (auto& it : blaze::subvector(Sel[tid], 0, index)) {
                    std::get<2>(it) /= std::get<1>(it);
                }
                std::nth_element( Sel[tid].begin()
                                , Sel[tid].begin() + (G_active - 1)
                                , Sel[tid].begin() + index
                                ,
                    [] (auto& lhs, auto& rhs)
                    -> bool {
                        return std::get<2>(lhs) > std::get<2>(rhs);
                    }
                );
                for (std::size_t g = 1; g < G_active; g++) {
                    GC(c, g) = std::get<0>(Sel[tid][g-1]);
                }
            } else {
                for (auto& it : blaze::subvector(blaze::row(GC, c), 0, G_active)) {
                    std::get<0>(Tab[tid][it]) = c;
                }
                for (auto& it : blaze::subvector(blaze::row(GC, c), 0, G_active)) {
                    if (Mask[it]) {
                        std::uniform_int_distribution<std::size_t> dst(0,C-1);
                        std::size_t index;
                        do {
                            index = dst(e[tid]);
                        }
                        while(Mask[index] || std::get<0>(Tab[tid][index]) == c);
                        std::get<0>(Tab[tid][index]) = c;
                        it = index;
                    }
                }
            }
        }
    });
}

template <template <typename, bool> class Theta, typename T, bool TPB>
std::tuple<std::size_t, double>
Base<Theta, T, TPB>::E_step(bool guess_variance)
{

    std::tuple<std::size_t, double> tmp;
    
    if (K == C) {
        std::get<0>(tmp) = compute_all();
    } else {
        std::get<0>(tmp) = search();
    }


    if (guess_variance)
    {
        //
        // assumes variance = 1 was used for search() (!)
        //

        double guess = tmp_min_sum / (D * W_sum);


        for (std::size_t n = 0; n < N; n++) {
            for (auto& it : blaze::subvector(blaze::row(Q, n), 0, K_active)) {
                std::get<1>(it) -= C0[std::get<0>(it)];
                std::get<1>(it) -= - 0.5 * D * blaze::log(2.0 * M_PI);
                std::get<1>(it) /= - 0.5;
                std::get<1>(it) *= - 0.5 / guess;
                std::get<1>(it) += - 0.5 * D * blaze::log(2.0 * M_PI * guess);
                std::get<1>(it) += C0[std::get<0>(it)];
            }
        }
    }


    std::get<1>(tmp) = distribution();


    if (K != C) {
        estimate();
    }



    return tmp;
}

template <template <typename, bool> class Theta, typename T, bool TPB>
void
Base<Theta, T, TPB>::M_step(void)
{
    M_step_M();
    static_cast<Theta<T, TPB>*>(this)->M_step_S();
}

template <template <typename, bool> class Theta, typename T, bool TPB>
void
Base<Theta, T, TPB>::M_step_M(void)
{
    blaze::DynamicVector<blaze::DynamicMatrix<T, blaze::rowMajor>> M_(
        adaptor.size(), 
        blaze::DynamicMatrix<T, blaze::rowMajor>(C, D, 0.)
    );
    blaze::DynamicVector<blaze::DynamicVector<double>> U_(
        adaptor.size(), 
        blaze::DynamicVector<double>(C, 0.)
    );

    adaptor.parallel(N, [&] (std::size_t n, std::size_t tid)
    -> void {
        for (const auto& it : blaze::subvector(blaze::row(Q, n), 0, K_active))
        {
            blaze::row(M_[tid], std::get<0>(it)) += W[n] * std::get<1>(it) * blaze::row(Y, n);
            U_[tid][std::get<0>(it)] += W[n] * std::get<1>(it);
        }
    });

    for (std::size_t tid = 1; tid < adaptor.size(); tid++)
    {
        M_[0] += M_[tid];
        U_[0] += U_[tid];
    }

    for (std::size_t c(0); c < C; c++)
    {
        if (!Mask[c]) {
            if (U_[0][c] > 0)
            {
                blaze::row(M_[0], c) /= U_[0][c];
                for (std::size_t d = 0; d < D; d++) {
                    if (!std::isfinite(M_[0](c, d))) {
                        blaze::row(M_[0], c) = blaze::row(M, c);
                        break;
                    }
                }
            } else {
                blaze::row(M_[0], c) = blaze::row(M, c);
            }  
        }
    }

    blaze::swap(M_[0], M);

    if (use_fixed_priors) {
        P = 1.0 / C_active;
    } else {
        double sum = 0;
        for (std::size_t c = 0; c < C; c++) {
            if (!Mask[c]) {
                sum += U_[0][c];
            }
        }
        for (std::size_t c = 0; c < C; c++) {
            if (!Mask[c]) {
                P[c] = U_[0][c] / sum;
                if (P[c] > 0) {
                    C0[c] = blaze::log(P[c]);
                    if (!std::isfinite(C0[c])) {
                        Mask[c] = true;
                    }
                } else {
                    Mask[c] = true;
                }
            }
        }
    }
}

template <template <typename, bool> class Theta, typename T, bool TPB>
void
Base<Theta, T, TPB>::update(void)
{
    C_active = C - blaze::nonZeros(Mask);
    K_active = std::min(K, C_active);
    G_active = std::min(G, C_active);
}

//--------------------------------------------------------------------------------------------------

template <typename T, bool TPB>
double
shared_variance_update( const blaze::DynamicMatrix<T, blaze::rowMajor>& Y
                      , const blaze::DynamicVector<double>& W
                      , double W_sum
                      , const blaze::DynamicMatrix<T, blaze::rowMajor>& M
                      , const blaze::DynamicMatrix<std::tuple<std::size_t, double>>& Q
                      , tp<TPB>& adaptor
                      , std::size_t N
                      , std::size_t D
                      , std::size_t K_active
) {
    blaze::DynamicVector<std::array<double, 64>> S_(
        adaptor.size(), 
        std::array<double, 64>{}
    );

    adaptor.parallel(N, [&] (std::size_t n, std::size_t tid)
    -> void {
        for (const auto& it : blaze::subvector(blaze::row(Q, n), 0, K_active))
        {
            S_[tid][0] += W[n] * std::get<1>(it) * blaze::sqrNorm(blaze::row(M, std::get<0>(it)) - blaze::row(Y, n));
        }
    });

    for (std::size_t tid = 1; tid < adaptor.size(); tid++)
    {
        S_[0][0] += S_[tid][0];
    }
    
    S_[0][0] /= W_sum * D;
    return S_[0][0];
}

//--------------------------------------------------------------------------------------------------

template <typename T, bool TPB> 
struct Shared : public Base<Shared, T, TPB>
{
    double& S;
    double C1;
    double C2;

    void constants(void)
    {
        C1 = - 0.5 * Base<Shared, T, TPB>::D * blaze::log(2.0 * M_PI * S);
        C2 = - 0.5 / S;
    }

    Shared( const blaze::DynamicMatrix<T, blaze::rowMajor>& Y_
          , const blaze::DynamicVector<double>& W_
          ,       blaze::DynamicMatrix<T, blaze::rowMajor>& M_
          ,       blaze::DynamicVector<double>& P_
          , std::size_t K_
          , std::size_t G_
          , bool R_
          , std::vector<std::mt19937_64>& e_
          , double& S_
          , tp<TPB>& adaptor_
    )
        : Base<Shared, T, TPB>(Y_, W_, M_, P_, K_, G_, R_, e_, adaptor_)
        , S(S_)
    {
        constants();
    }

    T
    density(std::size_t n, std::size_t c, double& tmp) const
    {
        tmp = blaze::sqrNorm(blaze::row(Base<Shared, T, TPB>::M, c) - blaze::row(Base<Shared,T, TPB>::Y, n));
        return Base<Shared, T, TPB>::C0[c] + C1 + C2 * tmp;
    }

    void M_step_S(void)
    {
        double S_ = shared_variance_update( Base<Shared, T, TPB>::Y
                                     , Base<Shared, T, TPB>::W
                                     , Base<Shared, T, TPB>::W_sum
                                     , Base<Shared, T, TPB>::M
                                     , Base<Shared, T, TPB>::Q
                                     , Base<Shared, T, TPB>::adaptor
                                     , Base<Shared, T, TPB>::N
                                     , Base<Shared, T, TPB>::D
                                     , Base<Shared, T, TPB>::K_active
        );
        S = S_;
        constants();
    }
};

//--------------------------------------------------------------------------------------------------

template <typename T, bool TPB> 
struct Iso : public Base<Iso, T, TPB>
{
    blaze::DynamicVector<double>& S;
    blaze::DynamicVector<double>C1;
    blaze::DynamicVector<double>C2;

    void constants(void)
    {
        for (std::size_t c(0); c < Base<Iso, T, TPB>::C; c++) {
            if (!Base<Iso, T, TPB>::Mask[c]) {
                C1[c] = - 0.5 * Base<Iso, T, TPB>::D * blaze::log(2.0 * M_PI * S[c]);
                C2[c] = - 0.5 / S[c];

                if (!std::isfinite(C1[c])) {
                    Base<Iso, T, TPB>::Mask[c] = true;
                }
                if (!std::isfinite(C2[c]) || (S[c] < Base<Iso, T, TPB>::threshold)) {
                    Base<Iso, T, TPB>::Mask[c] = true;
                }
            }
        }
    }

    Iso( const blaze::DynamicMatrix<T, blaze::rowMajor>& Y_
       , const blaze::DynamicVector<double>& W_
       ,       blaze::DynamicMatrix<T, blaze::rowMajor>& M_
       ,       blaze::DynamicVector<double>& P_
       , std::size_t K_
       , std::size_t G_
       , bool R_
       , std::vector<std::mt19937_64>& e_
       , blaze::DynamicVector<double>& S_
       , tp<TPB>& adaptor_
    )
        : Base<Iso, T, TPB>(Y_, W_, M_, P_, K_, G_, R_, e_, adaptor_)
        , S(S_)
        , C1(blaze::size(S_))
        , C2(blaze::size(S_))
    {
        for (std::size_t i(0); i < blaze::size(S); i++) {
            if (!(S[i] > 0)) {
                Base<Iso, T, TPB>::Mask[i] = true;
            }
        }
        constants();
        Base<Iso, T, TPB>::update();
    }

    double
    density(std::size_t n, std::size_t c, double& tmp) const
    {
        tmp = blaze::sqrNorm(blaze::row(Base<Iso, T, TPB>::M, c) - blaze::row(Base<Iso,T, TPB>::Y, n));
        double res = Base<Iso, T, TPB>::C0[c] + C1[c] + C2[c] * tmp;

        return res;
    }

    void M_step_S(void)
    {
        if (Base<Iso, T, TPB>::iso_update)
        {
            double S_ = shared_variance_update( Base<Iso, T, TPB>::Y
                                         , Base<Iso, T, TPB>::W
                                         , Base<Iso, T, TPB>::W_sum
                                         , Base<Iso, T, TPB>::M
                                         , Base<Iso, T, TPB>::Q
                                         , Base<Iso, T, TPB>::adaptor
                                         , Base<Iso, T, TPB>::N
                                         , Base<Iso, T, TPB>::D
                                         , Base<Iso, T, TPB>::K_active
            );
            S = S_;
        } else {

            blaze::DynamicVector<blaze::DynamicVector<double>> S_(
                Base<Iso, T, TPB>::adaptor.size(), 
                blaze::DynamicVector<double>(Base<Iso, T, TPB>::C, 0.)
            );
            blaze::DynamicVector<blaze::DynamicVector<double>> U_(
                Base<Iso, T, TPB>::adaptor.size(), 
                blaze::DynamicVector<double>(Base<Iso, T, TPB>::C, 0.)
            );

            Base<Iso, T, TPB>::adaptor.parallel(Base<Iso, T, TPB>::N, [&] (std::size_t n, std::size_t tid)
            -> void {
                for (const auto& it : blaze::subvector(blaze::row(Base<Iso, T, TPB>::Q, n), 0, Base<Iso, T, TPB>::K_active))
                {
                    S_[tid][std::get<0>(it)] += Base<Iso, T, TPB>::W[n] * std::get<1>(it) * blaze::sqrNorm(blaze::row(Base<Iso, T, TPB>::M, std::get<0>(it)) - blaze::row(Base<Iso, T, TPB>::Y, n));
                    U_[tid][std::get<0>(it)] += Base<Iso, T, TPB>::W[n] * std::get<1>(it);
                }
            });

            for (std::size_t tid = 1; tid < Base<Iso, T, TPB>::adaptor.size(); tid++) {
                S_[0] += S_[tid];
                U_[0] += U_[tid];
            }

            for (std::size_t c(0); c < Base<Iso, T, TPB>::C; c++) {
                if (!Base<Iso, T, TPB>::Mask[c]) {
                    if (U_[0][c] > 0) {
                        S_[0][c] /= U_[0][c] * Base<Iso, T, TPB>::D;
                        if (!std::isfinite(S_[0][c])) {
                            Base<Iso, T, TPB>::Mask[c] = true;
                        }
                        if (!(S_[0][c] > 0)) {
                            Base<Iso, T, TPB>::Mask[c] = true;
                        }
                    } else {
                        Base<Iso, T, TPB>::Mask[c] = true;
                    }
                }
            }

            blaze::swap(S_[0], S);
        }
        constants();
        Base<Iso, T, TPB>::update();
    }
};

//--------------------------------------------------------------------------------------------------

template <typename T, bool TPB> 
struct Dia : public Base<Dia, T, TPB>
{
    blaze::DynamicMatrix<T, blaze::rowMajor>& S;
    blaze::DynamicVector<T> C1;
    blaze::DynamicMatrix<T, blaze::rowMajor> C2;

    void constants(void)
    {
        for (std::size_t c(0); c < Base<Dia, T, TPB>::C; c++) {
            if (!Base<Dia, T, TPB>::Mask[c]) {
                C1[c] = - 0.5 * (Base<Dia, T, TPB>::D * blaze::log(2.0 * M_PI) + blaze::sum(blaze::log(blaze::row(S, c))));
                blaze::row(C2, c) = - 0.5 * blaze::row(S, c);
                for (std::size_t d(0); d < Base<Dia, T, TPB>::D; d++) {
                    C2(c, d) = - 0.5 / S(c, d);

                    if (!std::isfinite(C2(c,d)) || (S(c,d) < Base<Dia, T, TPB>::threshold)) {
                        Base<Dia, T, TPB>::Mask[c] = true;
                        break;
                    }
                }
                if (!std::isfinite(C1[c])) {
                    Base<Dia, T, TPB>::Mask[c] = true;
                }
                for (std::size_t d = 0; d < Base<Dia, T, TPB>::D; d++) {
                    if (!std::isfinite(C2(c, d))) {
                        Base<Dia, T, TPB>::Mask[c] = true;
                        break;
                    }
                }
            }
        }
    }

    Dia( const blaze::DynamicMatrix<T, blaze::rowMajor>& Y_
       , const blaze::DynamicVector<double>& W_
       ,       blaze::DynamicMatrix<T, blaze::rowMajor>& M_
       ,       blaze::DynamicVector<double>& P_
       , std::size_t K_
       , std::size_t G_
       , bool R_
       , std::vector<std::mt19937_64>& e_
       , blaze::DynamicMatrix<T, blaze::rowMajor>& S_
       , tp<TPB>& adaptor_
    )
        : Base<Dia, T, TPB>(Y_, W_, M_, P_, K_, G_, R_, e_, adaptor_)
        , S(S_)
        , C1(blaze::rows(S_))
        , C2(blaze::rows(S_), blaze::columns(S_))
    {
        for (std::size_t i(0); i < blaze::rows(S); i++) {
            if (blaze::min(blaze::row(S, i)) <= 0) {
                Base<Dia, T, TPB>::Mask[i] = true;
            }
        }
        constants();
        Base<Dia, T, TPB>::update();
    }

    double
    density(std::size_t n, std::size_t c, double& tmp) const
    {
        double tmp2 = blaze::dot((blaze::row(Base<Dia, T, TPB>::M, c) - blaze::row(Base<Dia,T, TPB>::Y, n)), blaze::row(C2, c) * (blaze::row(Base<Dia, T, TPB>::M, c) - blaze::row(Base<Dia,T, TPB>::Y, n)));
        tmp = (-2.0) * tmp2;
        double res = Base<Dia, T, TPB>::C0[c] + C1[c] + tmp2;

        return res;
    }

    void M_step_S(void)
    {
        if (Base<Dia, T, TPB>::iso_update)
        {
            double S_ = shared_variance_update( Base<Dia, T, TPB>::Y
                                         , Base<Dia, T, TPB>::W
                                         , Base<Dia, T, TPB>::W_sum
                                         , Base<Dia, T, TPB>::M
                                         , Base<Dia, T, TPB>::Q
                                         , Base<Dia, T, TPB>::adaptor
                                         , Base<Dia, T, TPB>::N
                                         , Base<Dia, T, TPB>::D
                                         , Base<Dia, T, TPB>::K_active
            );
            S = S_;
        } else {
            blaze::DynamicMatrix<T, blaze::rowMajor> S_(Base<Dia, T, TPB>::C, Base<Dia, T, TPB>::D, 0.);
            blaze::DynamicVector<double> U_(Base<Dia, T, TPB>::C, 0.);

            for (std::size_t n(0); n < Base<Dia, T, TPB>::N; n++)
            {
                for (const auto& it : blaze::subvector(blaze::row(Base<Dia, T, TPB>::Q, n), 0, Base<Dia, T, TPB>::K_active))
                {
                    blaze::row(S_, std::get<0>(it)) += Base<Dia, T, TPB>::W[n] * std::get<1>(it) * ((blaze::row(Base<Dia, T, TPB>::M, std::get<0>(it)) - blaze::row(Base<Dia, T, TPB>::Y, n)) * (blaze::row(Base<Dia, T, TPB>::M, std::get<0>(it)) - blaze::row(Base<Dia, T, TPB>::Y, n)));
                    U_[std::get<0>(it)] += Base<Dia, T, TPB>::W[n] * std::get<1>(it);
                }
            }

            for (std::size_t c(0); c < Base<Dia, T, TPB>::C; c++) {
                if (!Base<Dia, T, TPB>::Mask[c]) {
                    if (U_[c] > 0) {
                        blaze::row(S_, c) /= U_[c];
                        for (std::size_t d = 0; d < Base<Dia, T, TPB>::D; d++) {
                            if (!std::isfinite(S_(c, d))) {
                                Base<Dia, T, TPB>::Mask[c] = true;
                                break;
                            }
                        }
                        if (!(blaze::min(blaze::row(S_, c)) > 0)) {
                            Base<Dia, T, TPB>::Mask[c] = true;
                        }
                    } else {
                        Base<Dia, T, TPB>::Mask[c] = true;
                    }
                }
            }
            blaze::swap(S_, S);
        }
        constants();
        Base<Dia, T, TPB>::update();
    }
};

#endif
