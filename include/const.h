/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef CONST_H
#define CONST_H

#include "type.h"

namespace libheom {

#ifdef SUPPORT_GPU_PARALLELIZATION
const bool support_gpu_parallelization = true;
#else
const bool support_gpu_parallelization = false;
#endif

// Imaginary unit with given complex type
template <typename T>
inline constexpr T IUnit();
template <>
inline constexpr complex128 IUnit<complex128>() { return complex128(0.0 , 1.0 ); }
template <>
inline constexpr complex64  IUnit<complex64>()  { return complex64 (0.0f, 1.0f); }

// Zero with given type
template <typename T>
inline constexpr T Zero();
template <>
inline constexpr float32    Zero<float32>()    { return 0.0f; }
template <>
inline constexpr float64    Zero<float64>()    { return 0.0;  }
template <>
inline constexpr complex64  Zero<complex64>()  { return complex64 (Zero<float32>()); }
template <>
inline constexpr complex128 Zero<complex128>() { return complex128(Zero<float64>()); }

// One with given type
template <typename T>
inline constexpr T One();
template <>
inline constexpr float32    One<float32>()    { return 1.0f; }
template <>
inline constexpr float64    One<float64>()    { return 1.0;  }
template <>
inline constexpr complex64  One<complex64>()  { return complex64 (One<float32>()); }
template <>
inline constexpr complex128 One<complex128>() { return complex128(One<float64>()); }

// Fraction constant with given complex type
template <typename T>
inline constexpr T Frac(int num, int denom);
template <>
inline constexpr float32    Frac<float32>    (int num, int den) {
  return static_cast<float32>(num)/static_cast<float32>(den);
}
template <>
inline constexpr float64    Frac<float64>    (int num, int den) {
  return static_cast<float64>(num)/static_cast<float64>(den);
}
template <>
inline constexpr complex64  Frac<complex64> (int num, int den) {
  return complex64(Frac<float32>(num, den));
}
template <>
inline constexpr complex128 Frac<complex128>(int num, int den) {
  return complex128(Frac<float64>(num, den));
}

}
#endif /* CONST_H */
