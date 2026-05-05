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
constexpr Eigen::StorageOptions eigen_order = Eigen::RowMajor;

// inline prevents multiple-definition link errors (same as type.h align_val);
// removing "enum" keyword avoids nvcc rejecting "constexpr enum" in .cu files.
template<>
inline constexpr Eigen::StorageOptions eigen_order<row_major> = Eigen::RowMajor;
template<>
inline constexpr Eigen::StorageOptions eigen_order<col_major> = Eigen::ColMajor;

};

#endif
