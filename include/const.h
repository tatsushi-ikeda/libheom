/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef LIBHEOM_CONST_H
#define LIBHEOM_CONST_H

#include "type.h"

namespace libheom {

// imaginary unit with given complex type
template<typename T>
inline constexpr T i_unit();

template<>
inline constexpr complex128 i_unit<complex128>() { return complex128(0.0, 1.0 ); }

template<>
inline constexpr complex64  i_unit<complex64>()  { return complex64 (0.0f, 1.0f); }

// zero with given type
template<typename T>
inline constexpr T zero();

template<>
inline constexpr float32    zero<float32>()    { return 0.0f; }

template<>
inline constexpr float64    zero<float64>()    { return 0.0;  }

template<>
inline constexpr complex64  zero<complex64>()  { return complex64 (zero<float32>()); }

template<>
inline constexpr complex128 zero<complex128>() { return complex128(zero<float64>()); }

// one with given type
template<typename T>
inline constexpr T one();

template<>
inline constexpr float32    one<float32>()    { return 1.0f; }

template<>
inline constexpr float64    one<float64>()    { return 1.0;  }

template<>
inline constexpr complex64  one<complex64>()  { return complex64 (one<float32>()); }

template<>
inline constexpr complex128 one<complex128>() { return complex128(one<float64>()); }

// fraction constant with given complex type
template<typename T>
inline constexpr T frac(int num, int denom);

template<>
inline constexpr float32    frac<float32>(int num, int den)
{
  return static_cast<float32>(num) / static_cast<float32>(den);
}

template<>
inline constexpr float64    frac<float64>(int num, int den)
{
  return static_cast<float64>(num) / static_cast<float64>(den);
}

template<>
inline constexpr complex64  frac<complex64>(int num, int den)
{
  return complex64(frac<float32>(num, den));
}

template<>
inline constexpr complex128 frac<complex128>(int num, int den)
{
  return complex128(frac<float64>(num, den));
}

} // namespace libheom
#endif /* CONST_H */
