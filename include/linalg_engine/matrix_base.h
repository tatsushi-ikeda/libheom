/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef MATRIX_H
#define MATRIX_H

#ifdef __INTEL_COMPILER
#include <aligned_new>
#endif
#include "const.h"

namespace libheom
{

constexpr bool row_major = true;
constexpr bool col_major = false;

template<bool order>
constexpr int shape_index = 0;

template<>
constexpr int shape_index<row_major> = 0;
template<>
constexpr int shape_index<col_major> = 1;


template<int num_level, typename dtype, bool order, typename linalg_engine>
class matrix_base
{};

}

#endif /* MATRIX_H */
