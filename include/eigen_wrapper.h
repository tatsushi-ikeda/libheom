/*
 * LibHEOM: Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef EIGEN_WRAPPER
#define EIGEN_WRAPPER

#include "type.h"

#include "blas_wrapper.h"
#include "dense_matrix.h"
#include "csr_matrix.h"

namespace libheom {

template<typename T, int N>
struct copy_impl<T, N, dense_matrix>
{
  static inline void func
  /**/(const dense_matrix<T, N>& x,
       dense_matrix<T, N>& y)
  {
    y.noalias() = x;
  }
};


template<typename T, int N>
struct copy_impl<T, N, dense_vector>
{
  static inline void func
  /**/(const dense_vector<T, N>& x,
       dense_vector<T, N>& y)
  {
    y.noalias() = x;
  }
};


template<typename T, typename U, int N>
struct scal_impl<T, U, N, dense_matrix>
{
  static inline void func
  /**/(T alpha,
       dense_matrix<U, N> x)
  {
    x.noalias() *= a;
  }
};


template<typename T, int N>
struct axpy_impl<T, N, dense_matrix>
{
  static inline void func
  /**/(T alpha,
       const dense_matrix<T, N>& x,
       dense_matrix<T, N>& y)
  {
    y.noalias() += a*x;
  }
};


// general matrix-vector multiplication
template<typename T, int NA, int NB, int NC>
struct gemv_impl<T, NA, dense_matrix, NB, dense_vector, NC, dense_vector>
{
  static inline void func
  /**/(T alpha, 
       const dense_matrix<T, NA>& A,
       const dense_vector<T, NB>& B,
       T beta,
       dense_vector<T, NC>& C)
  {
    C *= beta;
    C.noalias() = alpha*A*B;
  }
};


template<typename T, int NA, int NB, int NC>
struct gemv_impl<T, NA, csr_matrix, NB, dense_vector, NC, dense_vector>
{
  static inline void func
  /**/(T alpha, 
       const csr_matrix<T, NA>& A,
       const dense_vector<T, NB>& B,
       T beta,
       dense_vector<T, NC>& C)
  {
    C *= beta;
    C.noalias() = alpha*A*B;
  }
};


// // general matrix-matrix multiplication
// template<template <typename> class MatrixTypeA,
//          template <typename> class MatrixTypeB,
//          template <typename> class MatrixTypeC>
// struct GemmImpl<complex64, MatrixTypeA, MatrixTypeB, MatrixTypeC> {
//   static inline void func(complex64 alpha,
//                           const MatrixTypeA<complex64>& A,
//                           const MatrixTypeB<complex64>& B,
//                           complex64& beta,
//                           MatrixTypeC<complex64>& C) {
//     cblas_cgemm(CblasRowMajor,
//                 CblasNoTrans,
//                 CblasNoTrans,
//                 std::get<0>(A.shape), std::get<1>(B.shape), std::get<1>(A.shape),
//                 &alpha,
//                 A.Data(),
//                 std::get<0>(A.shape),
//                 B.Data(),
//                 std::get<0>(B.shape),
//                 &beta,
//                 C.Data(),
//                 std::get<0>(C.shape));
//   }
// };

// template<template <typename> class MatrixTypeA,
//          template <typename> class MatrixTypeB,
//          template <typename> class MatrixTypeC>
// struct GemmImpl<complex128, MatrixTypeA, MatrixTypeB, MatrixTypeC> {
//   static inline void func(complex128 alpha,
//                           const MatrixTypeA<complex128>& A,
//                           const MatrixTypeB<complex128>& B,
//                           complex128& beta,
//                           MatrixTypeC<complex128>& C) {
//     cblas_zgemm(CblasRowMajor,
//                 CblasNoTrans,
//                 CblasNoTrans,
//                 std::get<0>(A.shape),
//                 std::get<1>(B.shape),
//                 std::get<1>(A.shape),
//                 &alpha,
//                 A.Data(),
//                 std::get<0>(A.shape),
//                 B.Data(),
//                 std::get<0>(B.shape),
//                 &beta,
//                 C.Data(),
//                 std::get<0>(C.shape));
//   }
// };


// template<template <typename> class MatrixTypeB,
//          template <typename> class MatrixTypeC>
// struct GemmImpl<complex64, CsrMatrix, MatrixTypeB, MatrixTypeC> {
//   static inline void func(complex64 alpha,
//                           const CsrMatrix<complex64>& A,
//                           const MatrixTypeB<complex64>& B,
//                           complex64& beta,
//                           MatrixTypeC<complex64>& C) {
// #if __INTEL_MKL__ >= 12
//     struct matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL };
//     mkl_sparse_c_mm(SPARSE_OPERATION_NON_TRANSPOSE,
//                     alpha,
//                     // A_handle,
//                     A.p_aux->handle,
//                     descr,
//                     SPARSE_LAYOUT_ROW_MAJOR,
//                     B.Data(),
//                     std::get<1>(B.shape),
//                     std::get<0>(B.shape),
//                     beta,
//                     C.Data(),
//                     std::get<0>(C.shape));
// #else
//     // When zero-based indexing is employed, mkl_csrcmm assumes that B and C are stored in row-major order.
//     // See,
//     // https://stackoverflow.com/questions/26536785/intel-mkl-spareblas-mm-csr-one-based-indexing-not-working
//     mkl_ccsrmm("N",
//                const_cast<int*>(&std::get<0>(A.shape)),
//                const_cast<int*>(&std::get<1>(C.shape)),
//                const_cast<int*>(&std::get<1>(A.shape)),
//                &alpha,
//                "G  C",
//                const_cast<complex64*>(A.data.data()),
//                const_cast<int*>(A.indices.data()),
//                const_cast<int*>(A.indptrb.data()),
//                const_cast<int*>(A.indptre.data()),
//                const_cast<complex64*>(B.Data()),
//                const_cast<int*>(&std::get<0>(B.shape)),
//                &beta,
//                C.Data(),
//                &std::get<0>(C.shape));
// #endif
//   }
// };


// template<template <typename> class MatrixTypeB,
//          template <typename> class MatrixTypeC>
// struct GemmImpl<complex128, CsrMatrix, MatrixTypeB, MatrixTypeC> {
//   static inline void func(complex128 alpha,
//                           const CsrMatrix<complex128>& A,
//                           const MatrixTypeB<complex128>& B,
//                           complex128& beta,
//                           MatrixTypeC<complex128>& C) {
// #if __INTEL_MKL__ >= 12
//     struct matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL };
//     mkl_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE,
//                     alpha,
//                     // A_handle,
//                     A.p_aux->handle,
//                     descr,
//                     SPARSE_LAYOUT_ROW_MAJOR,
//                     B.Data(),
//                     std::get<1>(B.shape),
//                     std::get<0>(B.shape),
//                     beta,
//                     C.Data(),
//                     std::get<0>(C.shape));
// #else
//     // When zero-based indexing is employed, mkl_csrcmm assumes that B and C are stored in row-major order.
//     // See, 
//     // https://stackoverflow.com/questions/26536785/intel-mkl-spareblas-mm-csr-one-based-indexing-not-working
//     mkl_zcsrmm("N",
//                const_cast<int*>(&std::get<0>(A.shape)),
//                const_cast<int*>(&std::get<1>(C.shape)),
//                const_cast<int*>(&std::get<1>(A.shape)),
//                &alpha,
//                "G  C",
//                const_cast<complex128*>(A.data.data()),
//                const_cast<int*>(A.indices.data()),
//                const_cast<int*>(A.indptrb.data()),
//                const_cast<int*>(A.indptre.data()),
//                const_cast<complex128*>(B.Data()),
//                const_cast<int*>(&std::get<0>(B.shape)),
//                &beta,
//                C.Data(),
//                &std::get<0>(C.shape));
// #endif    
//     // std::cout << "->C:" << C << std::endl;
//   }
// };


// template<template <typename> class MatrixTypeA,
//          template <typename> class MatrixTypeC>
// struct GemmImpl<complex64, MatrixTypeA, CsrMatrix, MatrixTypeC> {
//   static inline void func(complex64 alpha,
//                           const MatrixTypeA<complex64>& A,
//                           const CsrMatrix<complex64>& B,
//                           complex64& beta,
//                           MatrixTypeC<complex64>& C) {
// #if __INTEL_MKL__ >= 12
//     struct matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL };
//     mkl_sparse_c_mm(SPARSE_OPERATION_TRANSPOSE,
//                     alpha,
//                     // B_handle,
//                     B.p_aux->handle,
//                     descr,
//                     SPARSE_LAYOUT_COLUMN_MAJOR,
//                     A.Data(),
//                     std::get<0>(A.shape),
//                     std::get<1>(A.shape),
//                     beta,
//                     C.Data(),
//                     std::get<1>(C.shape));
// #else
//     complex64 one  = static_cast<complex64>(1);

//     mkl_cimatcopy('R', 'T',
//                   std::get<0>(A.shape), std::get<1>(A.shape),
//                   one,
//                   const_cast<complex64*>(A.Data()),
//                   std::get<0>(A.shape), std::get<0>(A.shape));
//     mkl_cimatcopy('R', 'T',
//                   std::get<0>(C.shape), std::get<1>(C.shape),
//                   one,
//                   const_cast<complex64*>(C.Data()),
//                   std::get<0>(C.shape), std::get<0>(C.shape));
    
//     // Calculate transpose of AB by (B_T*A_T)
//     mkl_ccsrmm("T",
//                const_cast<int*>(&std::get<1>(B.shape)),
//                const_cast<int*>(&std::get<0>(A.shape)),
//                const_cast<int*>(&std::get<0>(B.shape)),
//                &alpha,
//                "G  C",
//                const_cast<complex64*>(B.data.data()),
//                const_cast<int*>(B.indices.data()),
//                const_cast<int*>(B.indptrb.data()),
//                const_cast<int*>(B.indptre.data()),
//                const_cast<complex64*>(A.Data()),
//                const_cast<int*>(&std::get<0>(B.shape)),
//                &beta,
//                C.Data(),
//                &std::get<1>(C.shape));
    
//     mkl_cimatcopy('R', 'T',
//                   std::get<1>(A.shape), std::get<0>(A.shape),
//                   one,
//                   const_cast<complex64*>(A.Data()),
//                   std::get<1>(A.shape), std::get<1>(A.shape));
//     mkl_cimatcopy('R', 'T',
//                   std::get<1>(C.shape), std::get<0>(C.shape),
//                   one,
//                   const_cast<complex64*>(C.Data()),
//                   std::get<1>(C.shape), std::get<1>(C.shape));
// #endif    
//   }
// };


// template<template <typename> class MatrixTypeA,
//          template <typename> class MatrixTypeC>
// struct GemmImpl<complex128, MatrixTypeA, CsrMatrix, MatrixTypeC> {
//   static inline void func(complex128 alpha,
//                           const MatrixTypeA<complex128>& A,
//                           const CsrMatrix<complex128>& B,
//                           complex128& beta,
//                           MatrixTypeC<complex128>& C) {
// #if __INTEL_MKL__ >= 12
//     struct matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL };
//     mkl_sparse_z_mm(SPARSE_OPERATION_TRANSPOSE,
//                     alpha,
//                     // B_handle,
//                     B.p_aux->handle,
//                     descr,
//                     SPARSE_LAYOUT_COLUMN_MAJOR,
//                     A.Data(),
//                     std::get<0>(A.shape),
//                     std::get<1>(A.shape),
//                     beta,
//                     C.Data(),
//                     std::get<1>(C.shape));
// #else
//     complex128 one  = static_cast<complex128>(1);

//     mkl_zimatcopy('R', 'T',
//                   std::get<0>(A.shape), std::get<1>(A.shape),
//                   one,
//                   const_cast<complex128*>(A.Data()),
//                   std::get<0>(A.shape), std::get<0>(A.shape));
//     mkl_zimatcopy('R', 'T',
//                   std::get<0>(C.shape), std::get<1>(C.shape),
//                   one,
//                   const_cast<complex128*>(C.Data()),
//                   std::get<0>(C.shape), std::get<0>(C.shape));
    
//     // Calculate transpose of AB by (B_T*A_T)
//     mkl_zcsrmm("T",
//                const_cast<int*>(&std::get<1>(B.shape)),
//                const_cast<int*>(&std::get<0>(A.shape)),
//                const_cast<int*>(&std::get<0>(B.shape)),
//                &alpha,
//                "G  C",
//                const_cast<complex128*>(B.data.data()),
//                const_cast<int*>(B.indices.data()),
//                const_cast<int*>(B.indptrb.data()),
//                const_cast<int*>(B.indptre.data()),
//                const_cast<complex128*>(A.Data()),
//                const_cast<int*>(&std::get<0>(B.shape)),
//                &beta,
//                C.Data(),
//                &std::get<1>(C.shape));
    
//     mkl_zimatcopy('R', 'T',
//                   std::get<1>(A.shape), std::get<0>(A.shape),
//                   one,
//                   const_cast<complex128*>(A.Data()),
//                   std::get<1>(A.shape), std::get<1>(A.shape));
//     mkl_zimatcopy('R', 'T',
//                   std::get<1>(C.shape), std::get<0>(C.shape),
//                   one,
//                   const_cast<complex128*>(C.Data()),
//                   std::get<1>(C.shape), std::get<1>(C.shape));
// #endif
//   }
// };

}
#endif /* EIGEN_WRAPPER */
