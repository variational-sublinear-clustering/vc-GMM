/******************************************************************************/
//
//	Copyright (C) 2018, Florian Hirschberger <florian.hirschberger@uol.de>
//
//	LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
//	ACADEMIC FREE LICENSE (AFL) v3.0.
//
/******************************************************************************/

#ifndef UTILITY_IO_HPP
#define UTILITY_IO_HPP

#include <stdexcept>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <type_traits>

#include "blaze/Blaze.h"


template <typename T>
void
loadtxt(
    blaze::DynamicMatrix<T, blaze::rowMajor>& x,
    const std::string& path
) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::invalid_argument("could not open dataset file : " + path);
    }
    std::string line;
    std::vector<std::vector<T>> buf;

    while (std::getline(ifs, line)) {
        std::istringstream iss{line};
        buf.push_back(std::vector<T>{});
        T el;
        while (iss >> el) {
            buf.back().push_back(el);
        }
        if (buf.back().size() != buf.front().size()) {
            throw std::invalid_argument("different row sizes");
        }
    }
    if (!buf.empty()) {

        size_t row = buf.size();
        size_t col = buf.front().size();

        x.resize(row, col);

        for (size_t i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
                x(i, j) = buf[i][j];
            }
        }
    }
}

// writes blaze matrix as a text file
template <typename T>
void
savetxt(
    const std::string& path,
    const blaze::DynamicMatrix<T, blaze::rowMajor>& x
) {
    std::ofstream ofs(path);

    ofs << std::scientific;

    for (size_t i = 0; i < x.rows(); i++) {
        for (size_t j = 0; j < x.columns(); j++) {
            if (std::is_floating_point<T>::value) {
                ofs << static_cast<double      >(x(i,j)) << " ";
            } else {
                ofs << static_cast<std::int64_t>(x(i,j)) << " ";
            }
        }
        ofs << "\n";
    }
}

template <typename T>
void
savetxt(
    const std::string& path,
    const blaze::DynamicVector<T>& x
) {
    std::ofstream ofs(path);

    ofs << std::scientific;
    for (size_t i = 0; i < x.size(); i++) {
        if (std::is_floating_point<T>::value) {
            ofs << static_cast<double      >(x[i]) << " ";
        } else {
            ofs << static_cast<std::int64_t>(x[i]) << " ";
        }
    }
}

#endif
