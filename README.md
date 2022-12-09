# vc-GMM

Here we provide an example program and C++ implementation of the vc-GMM algorithms (vc-GMM-iso, vc-GMM-isoflex and vc-GMM-diag).
Given a dataset in the form of a .txt file (with values separated by whitespaces and without comments or any other characters, e.g. as generated by numpy.savetxt(...)),
the example program in 'main.cpp' reads this dataset and applies the selected vc-GMM algorithm.
The path of the dataset and other parameters can be set by editing 'main.cpp'.

We also provide a toy dataset 'example-dataset-birch-5x5.txt' with N = 10000 and D = 2.
To compute a clustering for this toy dataset with vc-GMM-iso and 25 clusters, simply compile and run main.cpp as provided.
For other parameters such as G, see main.cpp.

To compile the program, download the blaze library version 3.6 (https://bitbucket.org/blaze-lib/blaze/downloads/) and place 
the folder blaze-3.6 in the 'code-example/include/' subdirectory.
Depending on the platform, the example program can then be compiled with e.g.

    g++ -DNDEBUG -std=c++14 -I./include/blaze-3.6/ -O3 -o example-program main.cpp -lpthread

from within the 'code-example' subdirectory.
For better performance it is recommended to enable vector instructions e.g. with -mavx if available on the machine.

The here provided source code of the sublinear clustering algorithms is intended for documentation and demonstrative purposes.

For detailed information please see the associated paper:

"A Variational EM Acceleration for Efficient Clustering at Very Large Scales", Florian Hirschberger*, Dennis Forster*, Jörg Lücke.  
<em>IEEE Transactions on Pattern Analysis and Machine Intelligence</em>, vol. 44, no. 12, pp. 9787-9801, 2022.  
*joint first authors. DOI:10.1109/TPAMI.2021.3133763, https://ieeexplore.ieee.org/document/9645327
