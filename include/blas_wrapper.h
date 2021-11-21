/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef BLAS_WRAPPER_H
#define BLAS_WRAPPER_H

namespace libheom {

// copy: y = x
template<typename T,
         template <typename> class matrix_type>
struct copy_impl {
  static inline void func(const matrix_type<T>& x,
                          matrix_type<T>& y);
};
template<typename T,
         template <typename> class matrix_type>
inline void copy(const matrix_type<T>& x,
                 matrix_type<T>& y) {
  copy_impl<T, matrix_type>::func(x, y);
}


// x = alpha*x
template<typename T,
         typename U,
         template <typename> class matrix_type>
struct scal_impl {
  static inline void func(T alpha,
                          matrix_type<U>& x);
};
template<typename T,
         typename U,
         template <typename> class matrix_type>
inline void scal(T alpha,
                 matrix_type<U>& x) {
  scal_impl<T, U, matrix_type>::func(alpha, x);
}


// y = alpha*x + y
template<typename T,
         template <typename> class matrix_type>
struct axpy_impl {
  static inline void func(T alpha,
                          const matrix_type<T>& x,
                          matrix_type<T>& y);
};
template<typename T,
         template <typename> class matrix_type>
inline void axpy(T alpha,
                 const matrix_type<T>& x,
                 matrix_type<T>& y) {
  axpy_impl<T, matrix_type>::func(alpha, x, y);
}


// general matrix-vector multiplication
template<typename T,
         template <typename> class matrix_type>
struct gemv_impl {
  static inline void func(T alpha,
                          const matrix_type<T>& A,
                          const T* B,
                          T beta,
                          T* C);
};
template<typename T,
         template <typename> class matrix_type>
inline void gemv(T alpha,
          const matrix_type<T>& A,
          const T* B,
          T beta,
          T* C) {
  gemv_impl<T, matrix_type>::func(alpha, A, B, beta, C);
}


// general matrix-matrix multiplication
template<typename T,
         template <typename> class matrix_type_a,
         template <typename> class matrix_type_b,
         template <typename> class matrix_type_c>
struct gemm_impl {
  static inline void func(T alpha,
                          const matrix_type_a<T>& A,
                          const matrix_type_b<T>& B,
                          T beta,
                          matrix_type_c<T>& C);
};
template<typename T,
         template <typename> class matrix_type_a,
         template <typename> class matrix_type_b,
         template <typename> class matrix_type_c>
inline void gemm(T alpha,
                 const matrix_type_a<T>& A,
                 const matrix_type_b<T>& B,
                 T beta,
                 matrix_type_c<T>& C) {
  gemm_impl<T, matrix_type_a, matrix_type_b, matrix_type_c>::func(alpha, A, B, beta, C);
}

}

#endif /* BLAS_WRAPPER_H */
