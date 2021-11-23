/*
 * LibHEOM, Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef TYPE_H
#define TYPE_H

#include <complex>

namespace libheom
{

typedef float  float32;
typedef std::complex<float> complex64;

typedef double float64;
typedef std::complex<double> complex128;

template<typename T>
using real_t = typename T::value_type;

}

#endif
