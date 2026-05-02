/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LICENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef LIBHEOM_INCLUDE_EIGEN_H
#define LIBHEOM_INCLUDE_EIGEN_H

#define EIGEN_NO_DEBUG
#define EIGEN_NO_CUDA
#define EIGEN_STRONG_INLINE INLINE
#define EIGEN_INITIALIZE_MATRICES_BY_NAN

#ifdef EIGEN_USE_MKL_ALL
#  include "include_mkl.h"
#endif

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>

namespace libheom
{

template<order_t order>
constexpr enum Eigen::StorageOptions eigen_order = Eigen::RowMajor;

template<>
constexpr enum Eigen::StorageOptions eigen_order<row_major> = Eigen::RowMajor;
template<>
constexpr enum Eigen::StorageOptions eigen_order<col_major> = Eigen::ColMajor;

};

#endif
