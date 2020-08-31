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
         template <typename> class MatrixType>
struct CopyImpl {
  static inline void func(const MatrixType<T>& x,
                          MatrixType<T>& y);
};
template<typename T,
         template <typename> class MatrixType>
inline void copy(const MatrixType<T>& x,
                 MatrixType<T>& y) {
  CopyImpl<T, MatrixType>::func(x, y);
}


// x = alpha*x
template<typename T,
         typename U,
         template <typename> class MatrixType>
struct ScalImpl {
  static inline void func(T alpha,
                          MatrixType<U>& x);
};
template<typename T,
         typename U,
         template <typename> class MatrixType>
inline void scal(T alpha,
                 MatrixType<U>& x) {
  ScalImpl<T, U, MatrixType>::func(alpha, x);
}


// y = alpha*x + y
template<typename T,
         template <typename> class MatrixType>
struct AxpyImpl {
  static inline void func(T alpha,
                          const MatrixType<T>& x,
                          MatrixType<T>& y);
};
template<typename T,
         template <typename> class MatrixType>
inline void axpy(T alpha,
                 const MatrixType<T>& x,
                 MatrixType<T>& y) {
  AxpyImpl<T, MatrixType>::func(alpha, x, y);
}


// general matrix-vector multiplication
template<typename T,
         template <typename> class MatrixType>
struct GemvImpl {
  static inline void func(T alpha,
                          const MatrixType<T>& A,
                          const T* B,
                          T beta,
                          T* C);
};
template<typename T,
         template <typename> class MatrixType>
inline void gemv(T alpha,
          const MatrixType<T>& A,
          const T* B,
          T beta,
          T* C) {
  GemvImpl<T, MatrixType>::func(alpha, A, B, beta, C);
}


// general matrix-matrix multiplication
template<typename T,
         template <typename> class MatrixTypeA,
         template <typename> class MatrixTypeB,
         template <typename> class MatrixTypeC>
struct GemmImpl {
  static inline void func(T alpha,
                          const MatrixTypeA<T>& A,
                          const MatrixTypeB<T>& B,
                          T beta,
                          MatrixTypeC<T>& C);
};
template<typename T,
         template <typename> class MatrixTypeA,
         template <typename> class MatrixTypeB,
         template <typename> class MatrixTypeC>
inline void gemm(T alpha,
                 const MatrixTypeA<T>& A,
                 const MatrixTypeB<T>& B,
                 T beta,
                 MatrixTypeC<T>& C) {
  GemmImpl<T, MatrixTypeA, MatrixTypeB, MatrixTypeC>::func(alpha, A, B, beta, C);
}

}

#endif /* BLAS_WRAPPER_H */
