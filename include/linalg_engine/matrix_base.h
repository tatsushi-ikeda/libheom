/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef LIBHEOM_MATRIX_H
#define LIBHEOM_MATRIX_H

#ifdef __INTEL_COMPILER
#include <aligned_new>
#endif
#include "const.h"

namespace libheom
{

using order_t = bool;
constexpr order_t row_major = true;
constexpr order_t col_major = false;

template<order_t order>
constexpr int shape_index = 0;

template<>
constexpr int shape_index<row_major> = 0;
template<>
constexpr int shape_index<col_major> = 1;

template<int num_level, typename dtype, order_t order, typename linalg_engine>
class matrix_base
{};

}

#endif
