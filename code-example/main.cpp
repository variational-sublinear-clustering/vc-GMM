/******************************************************************************/
//
//	Copyright (C) 2018, Florian Hirschberger <florian.hirschberger@uol.de>
//
//	LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
//	ACADEMIC FREE LICENSE (AFL) v3.0.
//
/******************************************************************************/

#include "include/fit.hpp"
#include "include/io.hpp"
#include "include/k-means-lwcs.hpp"

int main(void)
{
    //------------------------------------------------------------------------------------------------------------------
    //  set path to dataset and parameters

    std::string path         = "example-dataset-birch-5x5.txt";         // path to dataset as .txt file, e.g. use 
                                                                        // numpy.savetxt("<path-to-dataset>.txt", x)
                                                                        // with x.shape[0] = number of datapoints and
                                                                        // x.shape[1] = number of features
    std::size_t coreset_size = 1000;                                    // size of LWCS coreset
    std::size_t C            = 25;                                      // number of clusters
    std::size_t G            = 5;                                       // parameter G
    std::size_t chain_length = 20;                                      // chain length for AFK-MC^2 seeding
    bool        random       = false;                                   // use extra random cluster
    double      eps          = 1.e-4;                                   // threshold for convergence
    std::size_t nthreads     = std::thread::hardware_concurrency();     // number of threads
    std::size_t seed         = 1;                                       // random seed

    //------------------------------------------------------------------------------------------------------------------
    //  load dataset

    blaze::DynamicMatrix<double, blaze::rowMajor> Y;
    loadtxt(Y, path);

    std::cout << "number of datapoints = " << blaze::rows   (Y) << std::endl;
    std::cout << "number of features   = " << blaze::columns(Y) << std::endl;

    //------------------------------------------------------------------------------------------------------------------
    //  coreset construction

    blaze::DynamicMatrix<double, blaze::rowMajor> Y_core(coreset_size, blaze::columns(Y));  // coreset
    blaze::DynamicVector<double> W(coreset_size);                                           // weights

    std::mt19937_64 mt;
    mt.seed(seed);
    seed++;
    mt.discard(1e3);

    k_means_lwcs(Y, Y_core, W, mt);

    //  for the full dataset, i.e. v-GMM, use:
    //
    //  Y_core = Y;
    //  W.resize(blaze::rows(Y));
    //  W = 1.0;

    //------------------------------------------------------------------------------------------------------------------
    //  select v(c)-GMM algorithm

    blaze::DynamicMatrix<double, blaze::rowMajor> M(C, blaze::columns(Y));  // cluster means
    blaze::DynamicVector<double> P(C, 1.0/C);                               // priors

    //  for v(c)-GMM-diag, use:
    //
    //  blaze::DynamicMatrix<double, blaze::rowMajor> S(C, blaze::columns(Y));

    //  for v(c)-GMM-isoflex, use:
    //
    //  blaze::DynamicVector<double> S(C);

    //  for v(c)-GMM-iso, use:

    double S;

    //------------------------------------------------------------------------------------------------------------------
    //  run v(c)-GMM

    fit(Y_core, W, M, P, seed, S, G, G, random, chain_length, eps, nthreads);

    //  after completion of the algorithm, the parameters of the GMM are contained in
    //  M for the means, S for the variance(s) and P for the priors
    //
    //  To save the cluster means, e.g. do
    //
    //  savetxt("cluster-center.txt", M);

    return 0;
}