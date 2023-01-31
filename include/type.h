/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef LIBHEOM_TYPE_H
#define LIBHEOM_TYPE_H

#include <memory>
#include <complex>
#include <string>
#include <map>
#include <any>
#include <vector>

namespace libheom
{

using std::vector;

typedef float  float32;
typedef std::complex<float>  complex64;

typedef double float64;
typedef std::complex<double> complex128;

template<typename T>
using real_t = typename T::value_type;

template <typename dtype>
struct complex_type;

template<typename dtype>
using complex_t = typename complex_type<dtype>::value;

template <> struct complex_type<float32> { typedef complex64  value; };
template <> struct complex_type<float64> { typedef complex128 value; };

template<typename dtype>
constexpr int align_val = 0;

template <>
constexpr int align_val<complex64>  = 32;
template <>
constexpr int align_val<complex128> = 64;

using kwargs_t = std::map<std::string,std::any>;

}

#endif
