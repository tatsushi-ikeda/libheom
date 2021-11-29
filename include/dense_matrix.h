/*
 * LibHEOM: Copyright (c) Tatsushi Ikeda
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

namespace libheom
{

template<typename T>
using ref = Eigen::Ref<T,Eigen::Aligned64>;  // 


template<typename T,
         int N>
// using dense_vector = Eigen::Matrix<T,1,N,Eigen::RowMajor>;
using dense_vector = Eigen::Matrix<T,N,1,Eigen::ColMajor>;


template<typename T,
         int N>
// using dense_matrix = Eigen::Matrix<T,N,N,Eigen::RowMajor>;
using dense_matrix = Eigen::Matrix<T,N,N,Eigen::ColMajor>;


}
#endif /* DENSE_MATRIX_H */
