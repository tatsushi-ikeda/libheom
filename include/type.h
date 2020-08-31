/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef TYPE_H
#define TYPE_H

#include <complex>

namespace libheom {

typedef float  float32;
typedef std::complex<float> complex64;

typedef double float64;
typedef std::complex<double> complex128;

#define REAL_TYPE(T) typename T::value_type

}

#endif
