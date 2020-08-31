/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef MKL_WRAPPER_H
#define MKL_WRAPPER_H

#include "type.h"

#define MKL_Complex8  libheom::complex64
#define MKL_Complex16 libheom::complex128

#include <mkl.h>
#include <mkl_types.h>
#include <mkl_blas.h>
#include <mkl_spblas.h>


//#include <Eigen/Core>

#include "mathop.h"
#include "dense_matrix.h"
#include "csr_matrix.h"

namespace libheom {

template <typename T>
struct CsrMatrix<T>::Aux {
  bool init_flag = false;
#if __INTEL_MKL__ >= 12
  sparse_matrix_t handle;
#endif
  void Init(CsrMatrix<T>& mat);
};


template <typename T>
void CsrMatrix<T>::InitAux() {
  p_aux.reset(new CsrMatrix<T>::Aux());
  p_aux->Init(*this);
  p_aux->init_flag = true;
}


template <typename T>
void CsrMatrix<T>::FinAux() {
  if (p_aux && p_aux->init_flag) {
#if __INTEL_MKL__ >= 12
    mkl_sparse_destroy(p_aux->handle);
#endif
    p_aux->init_flag = false;
  }
  p_aux.reset();
}


template<>
struct CopyImpl<complex64> {
  static inline void func(int n,
                          const complex64* x,
                          complex64* y) {
    cblas_ccopy(n, x, 1, y, 1);
  }
};


template<>
struct CopyImpl<complex128> {
  static inline void func(int n,
                          const complex128* x,
                          complex128* y) {
    cblas_zcopy(n, x, 1, y, 1);
  }
};


template<>
struct ScalImpl<complex64, complex64> {
  static inline void func(int n, complex64 alpha, complex64* x) {
    cblas_cscal(n, &alpha, x, 1);
  }
};


template<>
struct ScalImpl<complex128, complex128> {
  static inline void func(int n, complex128 alpha, complex128* x) {
    cblas_zscal(n, &alpha, x, 1);
  }
};

template<>
struct ScalImpl<float32, complex64> {
  static inline void func(int n, float32 alpha, complex64* x) {
    cblas_csscal(n, alpha, x, 1);
  }
};


template<>
struct ScalImpl<float64, complex128> {
  static inline void func(int n, float64 alpha, complex128* x) {
    cblas_zdscal(n, alpha, x, 1);
  }
};


template<>
struct AxpyImpl<complex64> {
  static inline void func(int n,
                          complex64 alpha,
                          const complex64* x,
                          complex64* y) {
    cblas_caxpy(n, &alpha, x, 1, y, 1);
  }
};


template<>
struct AxpyImpl<complex128> {
  static inline void func(int n,
                          complex128 alpha,
                          const complex128* x,
                          complex128* y) {
    cblas_zaxpy(n, &alpha, x, 1, y, 1);
    // double alpha_ = std::real(alpha);
    // for (int i = 0; i < n; ++i) {
    //   y[i] += alpha_*x[i];
    // }
  }
};


// general matrix-vector multiplication
template<template <typename> class MatrixType>
struct GemvImpl<complex64, MatrixType> {
  static inline void func(complex64 alpha, 
                          const MatrixType<complex64>& A,
                          const complex64* B,
                          complex64& beta,
                          complex64* C) {
    cblas_cgemv(CblasRowMajor,
                CblasNoTrans,
                std::get<0>(A.shape),
                std::get<1>(A.shape),
                &alpha,
                A.Data(),
                std::get<0>(A.shape),
                B,
                1,
                &beta,
                C,
                1);
  }
};


template<template <typename> class MatrixType>
struct GemvImpl<complex128, MatrixType> {
  static inline void func(complex128 alpha, 
                          const MatrixType<complex128>& A,
                          const complex128* B,
                          complex128 beta,
                          complex128* C) {
    cblas_zgemv(CblasRowMajor,
                CblasNoTrans,
                std::get<0>(A.shape),
                std::get<1>(A.shape),
                &alpha,
                A.Data(),
                std::get<0>(A.shape),
                B,
                1,
                &beta,
                C,
                1);
    // Eigen::Map<const Eigen::Matrix<complex128,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>
    //     A_eigen(A.Data(), std::get<0>(A.shape), std::get<1>(A.shape));
    // Eigen::Map<const Eigen::Matrix<complex128,Eigen::Dynamic,1>>
    //     B_eigen(B, std::get<1>(A.shape));
    // Eigen::Map<Eigen::Matrix<complex128,Eigen::Dynamic,1>>
    //     C_eigen(C, std::get<0>(A.shape));
    // C_eigen.noalias() = alpha*A_eigen*B_eigen + beta*C_eigen;
    // int m = std::get<0>(A.shape);
    // int n = std::get<1>(A.shape);
    // double alpha_ = std::real(alpha);
    // double beta_ = std::real(beta);
    // auto A_data = A.Data();
    // for (int i = 0; i < m; ++i) {
    //   complex128 C_ = C[i]*beta_;
    //   for (int j = 0; j < n; ++j) {
    //     C_ += alpha_*A_data[i*n + j]*B[j];
    //   }
    //   C[i] = C_;
    // }
  }
};


template<>
struct GemvImpl<complex64, CsrMatrix> {
  static inline void func(complex64 alpha, 
                          const CsrMatrix<complex64>& A,
                          const complex64* B,
                          complex64& beta,
                          complex64* C) {
#if __INTEL_MKL__ >= 12
    struct matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL };
    mkl_sparse_c_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                    alpha,
                    A.p_aux->handle,
                    descr,
                    B,
                    beta,
                    C);
#else  
    mkl_ccsrmv("N",
               const_cast<int*>(&std::get<0>(A.shape)),
               const_cast<int*>(&std::get<1>(A.shape)),
               const_cast<complex64*>(&alpha),
               "G  C",
               const_cast<complex64*>(A.data.data()),
               const_cast<int*>(A.indices.data()),
               const_cast<int*>(A.indptrb.data()),
               const_cast<int*>(A.indptre.data()),
               const_cast<complex64*>(B),
               const_cast<complex64*>(&beta),
               C);
#endif  
  }
};


template<>
struct GemvImpl<complex128, CsrMatrix> {
  static inline void func(complex128 alpha, 
                          const CsrMatrix<complex128>& A,
                          const complex128* B,
                          complex128& beta,
                          complex128* C) {
#if __INTEL_MKL__ >= 12
    struct matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL };
    mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                    alpha,
                    A.p_aux->handle,
                    descr,
                    B,
                    beta,
                    C);
#else  
    mkl_zcsrmv("N",
               const_cast<int*>(&std::get<0>(A.shape)),
               const_cast<int*>(&std::get<1>(A.shape)),
               const_cast<complex128*>(&alpha),
               "G  C",
               const_cast<complex128*>(A.data.data()),
               const_cast<int*>(A.indices.data()),
               const_cast<int*>(A.indptrb.data()),
               const_cast<int*>(A.indptre.data()),
               const_cast<complex128*>(B),
               const_cast<complex128*>(&beta),
               C);
#endif
  }
};


// general matrix-matrix multiplication
template<template <typename> class MatrixTypeA,
         template <typename> class MatrixTypeB,
         template <typename> class MatrixTypeC>
struct GemmImpl<complex64, MatrixTypeA, MatrixTypeB, MatrixTypeC> {
  static inline void func(complex64 alpha,
                          const MatrixTypeA<complex64>& A,
                          const MatrixTypeB<complex64>& B,
                          complex64& beta,
                          MatrixTypeC<complex64>& C) {
    cblas_cgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                std::get<0>(A.shape), std::get<1>(B.shape), std::get<1>(A.shape),
                &alpha,
                A.Data(),
                std::get<0>(A.shape),
                B.Data(),
                std::get<0>(B.shape),
                &beta,
                C.Data(),
                std::get<0>(C.shape));
  }
};

template<template <typename> class MatrixTypeA,
         template <typename> class MatrixTypeB,
         template <typename> class MatrixTypeC>
struct GemmImpl<complex128, MatrixTypeA, MatrixTypeB, MatrixTypeC> {
  static inline void func(complex128 alpha,
                          const MatrixTypeA<complex128>& A,
                          const MatrixTypeB<complex128>& B,
                          complex128& beta,
                          MatrixTypeC<complex128>& C) {
    cblas_zgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                std::get<0>(A.shape),
                std::get<1>(B.shape),
                std::get<1>(A.shape),
                &alpha,
                A.Data(),
                std::get<0>(A.shape),
                B.Data(),
                std::get<0>(B.shape),
                &beta,
                C.Data(),
                std::get<0>(C.shape));
  }
};


template<template <typename> class MatrixTypeB,
         template <typename> class MatrixTypeC>
struct GemmImpl<complex64, CsrMatrix, MatrixTypeB, MatrixTypeC> {
  static inline void func(complex64 alpha,
                          const CsrMatrix<complex64>& A,
                          const MatrixTypeB<complex64>& B,
                          complex64& beta,
                          MatrixTypeC<complex64>& C) {
#if __INTEL_MKL__ >= 12
    struct matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL };
    mkl_sparse_c_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                    alpha,
                    // A_handle,
                    A.p_aux->handle,
                    descr,
                    SPARSE_LAYOUT_ROW_MAJOR,
                    B.Data(),
                    std::get<1>(B.shape),
                    std::get<0>(B.shape),
                    beta,
                    C.Data(),
                    std::get<0>(C.shape));
#else
    // When zero-based indexing is employed, mkl_csrcmm assumes that B and C are stored in row-major order.
    // See,
    // https://stackoverflow.com/questions/26536785/intel-mkl-spareblas-mm-csr-one-based-indexing-not-working
    mkl_ccsrmm("N",
               const_cast<int*>(&std::get<0>(A.shape)),
               const_cast<int*>(&std::get<1>(C.shape)),
               const_cast<int*>(&std::get<1>(A.shape)),
               &alpha,
               "G  C",
               const_cast<complex64*>(A.data.data()),
               const_cast<int*>(A.indices.data()),
               const_cast<int*>(A.indptrb.data()),
               const_cast<int*>(A.indptre.data()),
               const_cast<complex64*>(B.Data()),
               const_cast<int*>(&std::get<0>(B.shape)),
               &beta,
               C.Data(),
               &std::get<0>(C.shape));
#endif
  }
};


template<template <typename> class MatrixTypeB,
         template <typename> class MatrixTypeC>
struct GemmImpl<complex128, CsrMatrix, MatrixTypeB, MatrixTypeC> {
  static inline void func(complex128 alpha,
                          const CsrMatrix<complex128>& A,
                          const MatrixTypeB<complex128>& B,
                          complex128& beta,
                          MatrixTypeC<complex128>& C) {
#if __INTEL_MKL__ >= 12
    struct matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL };
    mkl_sparse_z_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                    alpha,
                    // A_handle,
                    A.p_aux->handle,
                    descr,
                    SPARSE_LAYOUT_ROW_MAJOR,
                    B.Data(),
                    std::get<1>(B.shape),
                    std::get<0>(B.shape),
                    beta,
                    C.Data(),
                    std::get<0>(C.shape));
#else
    // When zero-based indexing is employed, mkl_csrcmm assumes that B and C are stored in row-major order.
    // See, 
    // https://stackoverflow.com/questions/26536785/intel-mkl-spareblas-mm-csr-one-based-indexing-not-working
    mkl_zcsrmm("N",
               const_cast<int*>(&std::get<0>(A.shape)),
               const_cast<int*>(&std::get<1>(C.shape)),
               const_cast<int*>(&std::get<1>(A.shape)),
               &alpha,
               "G  C",
               const_cast<complex128*>(A.data.data()),
               const_cast<int*>(A.indices.data()),
               const_cast<int*>(A.indptrb.data()),
               const_cast<int*>(A.indptre.data()),
               const_cast<complex128*>(B.Data()),
               const_cast<int*>(&std::get<0>(B.shape)),
               &beta,
               C.Data(),
               &std::get<0>(C.shape));
#endif    
    // std::cout << "->C:" << C << std::endl;
  }
};


template<template <typename> class MatrixTypeA,
         template <typename> class MatrixTypeC>
struct GemmImpl<complex64, MatrixTypeA, CsrMatrix, MatrixTypeC> {
  static inline void func(complex64 alpha,
                          const MatrixTypeA<complex64>& A,
                          const CsrMatrix<complex64>& B,
                          complex64& beta,
                          MatrixTypeC<complex64>& C) {
#if __INTEL_MKL__ >= 12
    struct matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL };
    mkl_sparse_c_mm(SPARSE_OPERATION_TRANSPOSE,
                    alpha,
                    // B_handle,
                    B.p_aux->handle,
                    descr,
                    SPARSE_LAYOUT_COLUMN_MAJOR,
                    A.Data(),
                    std::get<0>(A.shape),
                    std::get<1>(A.shape),
                    beta,
                    C.Data(),
                    std::get<1>(C.shape));
#else
    complex64 one  = static_cast<complex64>(1);

    mkl_cimatcopy('R', 'T',
                  std::get<0>(A.shape), std::get<1>(A.shape),
                  one,
                  const_cast<complex64*>(A.Data()),
                  std::get<0>(A.shape), std::get<0>(A.shape));
    mkl_cimatcopy('R', 'T',
                  std::get<0>(C.shape), std::get<1>(C.shape),
                  one,
                  const_cast<complex64*>(C.Data()),
                  std::get<0>(C.shape), std::get<0>(C.shape));
    
    // Calculate transpose of AB by (B_T*A_T)
    mkl_ccsrmm("T",
               const_cast<int*>(&std::get<1>(B.shape)),
               const_cast<int*>(&std::get<0>(A.shape)),
               const_cast<int*>(&std::get<0>(B.shape)),
               &alpha,
               "G  C",
               const_cast<complex64*>(B.data.data()),
               const_cast<int*>(B.indices.data()),
               const_cast<int*>(B.indptrb.data()),
               const_cast<int*>(B.indptre.data()),
               const_cast<complex64*>(A.Data()),
               const_cast<int*>(&std::get<0>(B.shape)),
               &beta,
               C.Data(),
               &std::get<1>(C.shape));
    
    mkl_cimatcopy('R', 'T',
                  std::get<1>(A.shape), std::get<0>(A.shape),
                  one,
                  const_cast<complex64*>(A.Data()),
                  std::get<1>(A.shape), std::get<1>(A.shape));
    mkl_cimatcopy('R', 'T',
                  std::get<1>(C.shape), std::get<0>(C.shape),
                  one,
                  const_cast<complex64*>(C.Data()),
                  std::get<1>(C.shape), std::get<1>(C.shape));
#endif    
  }
};


template<template <typename> class MatrixTypeA,
         template <typename> class MatrixTypeC>
struct GemmImpl<complex128, MatrixTypeA, CsrMatrix, MatrixTypeC> {
  static inline void func(complex128 alpha,
                          const MatrixTypeA<complex128>& A,
                          const CsrMatrix<complex128>& B,
                          complex128& beta,
                          MatrixTypeC<complex128>& C) {
#if __INTEL_MKL__ >= 12
    struct matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL };
    mkl_sparse_z_mm(SPARSE_OPERATION_TRANSPOSE,
                    alpha,
                    // B_handle,
                    B.p_aux->handle,
                    descr,
                    SPARSE_LAYOUT_COLUMN_MAJOR,
                    A.Data(),
                    std::get<0>(A.shape),
                    std::get<1>(A.shape),
                    beta,
                    C.Data(),
                    std::get<1>(C.shape));
#else
    complex128 one  = static_cast<complex128>(1);

    mkl_zimatcopy('R', 'T',
                  std::get<0>(A.shape), std::get<1>(A.shape),
                  one,
                  const_cast<complex128*>(A.Data()),
                  std::get<0>(A.shape), std::get<0>(A.shape));
    mkl_zimatcopy('R', 'T',
                  std::get<0>(C.shape), std::get<1>(C.shape),
                  one,
                  const_cast<complex128*>(C.Data()),
                  std::get<0>(C.shape), std::get<0>(C.shape));
    
    // Calculate transpose of AB by (B_T*A_T)
    mkl_zcsrmm("T",
               const_cast<int*>(&std::get<1>(B.shape)),
               const_cast<int*>(&std::get<0>(A.shape)),
               const_cast<int*>(&std::get<0>(B.shape)),
               &alpha,
               "G  C",
               const_cast<complex128*>(B.data.data()),
               const_cast<int*>(B.indices.data()),
               const_cast<int*>(B.indptrb.data()),
               const_cast<int*>(B.indptre.data()),
               const_cast<complex128*>(A.Data()),
               const_cast<int*>(&std::get<0>(B.shape)),
               &beta,
               C.Data(),
               &std::get<1>(C.shape));
    
    mkl_zimatcopy('R', 'T',
                  std::get<1>(A.shape), std::get<0>(A.shape),
                  one,
                  const_cast<complex128*>(A.Data()),
                  std::get<1>(A.shape), std::get<1>(A.shape));
    mkl_zimatcopy('R', 'T',
                  std::get<1>(C.shape), std::get<0>(C.shape),
                  one,
                  const_cast<complex128*>(C.Data()),
                  std::get<1>(C.shape), std::get<1>(C.shape));
#endif
  }
};

}
#endif /* MKL_WRAPPER_H */
