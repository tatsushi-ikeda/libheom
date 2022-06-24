/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef INCLUDE_EIGEN_H
#define INCLUDE_EIGEN_H

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

// note: below codes are necessary when this file is compiled with old eigen

// #if (defined __NVCC__ && defined __ICC)
// #  include <icc/immintrin.h>
// #  define _CMP_EQ_OQ     _MM_CMPINT_EQ     // 0x00
// #  define _CMP_LT_OS     _MM_CMPINT_LT     // 0x01
// #  define _CMP_LE_OS     _MM_CMPINT_LE     // 0x02
// #  define _CMP_UNORD_Q   _MM_CMPINT_UNUSED // 0x03
// #  define _CMP_NEQ_UQ    _MM_CMPINT_NE     // 0x04
// #  define _CMP_NLT_US    _MM_CMPINT_NLT    // 0x05
// #  define _CMP_NLE_US    _MM_CMPINT_NLE    // 0x06
// #endif

namespace libheom
{

template<bool order>
constexpr enum Eigen::StorageOptions eigen_order = Eigen::RowMajor;

template<>
constexpr enum Eigen::StorageOptions eigen_order<row_major> = Eigen::RowMajor;
template<>
constexpr enum Eigen::StorageOptions eigen_order<col_major> = Eigen::ColMajor;

};

#endif
