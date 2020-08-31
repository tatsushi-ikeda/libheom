/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef DENSE_MATRIX_H
#define DENSE_MATRIX_H

#include <limits>
#include <tuple>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#define EIGEN_INITIALIZE_MATRICES_BY_NAN
#include <Eigen/Core>

namespace libheom {

template<typename T>
using Ref = Eigen::Ref<T>;

template<typename T,
         int N>
using DenseVector = Eigen::Matrix<T,N,1,Eigen::ColMajor>;

template<typename T,
         int N>
using DenseMatrix = Eigen::Matrix<T,N,N,Eigen::ColMajor>;


}
#endif /* DENSE_MATRIX_H */
