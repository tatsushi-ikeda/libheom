/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LICENSE.txt for license.
 *------------------------------------------------------------------------*/

#ifndef LIBHEOM_INCLUDE_MKL_H
#define LIBHEOM_INCLUDE_MKL_H

#include "type.h"

#define MKL_Complex8  libheom::complex64
#define MKL_Complex16 libheom::complex128

#include <mkl.h>

namespace libheom {

template <typename dtype>
struct mkl_type;

template<typename dtype>
using mkl_t = typename mkl_type<dtype>::value;

template <> struct mkl_type<complex64>  { typedef MKL_Complex8 value; };
template <> struct mkl_type<complex128> { typedef MKL_Complex16 value; };

}

#endif
