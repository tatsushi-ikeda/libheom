/*
 * LibHEOM: Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef CUBLAS_CUSPARSE_WRAPPER_H
#define CUBLAS_CUSPARSE_WRAPPER_H

#include "type_gpu.h"
#include "handle_gpu.h"
#include "dense_matrix_gpu.h"

namespace libheom {

// general matrix-vector multiplication
template<typename T,
         template <typename> class matrix_type_gpu>
struct gemv_impl_gpu {
  static void func(handle_gpu& handle,
                   T alpha,
                   const matrix_type_gpu<T>& A,
                   const thrust::device_vector<GPU_TYPE(T)>& B,
                   T beta,
                   thrust::device_vector<GPU_TYPE(T)>& C);
};

template<typename T,
         template <typename> class matrix_type_gpu>
void gemvGpu(handle_gpu& handle,
             T alpha,
             const matrix_type_gpu<T>& A,
             const thrust::device_vector<GPU_TYPE(T)>& B,
             T beta,
             thrust::device_vector<GPU_TYPE(T)>& C) {
  gemv_impl_gpu<T, matrix_type_gpu>::func(
      handle, alpha, A, B, beta, C);
}


template<template <typename> class matrix_type_gpu>
struct gemv_impl_gpu<complex64, matrix_type_gpu> {
  static void func(handle_gpu& handle,
                   complex64 alpha,
                   const matrix_type_gpu<complex64>& A,
                   const thrust::device_vector<GPU_TYPE(complex64)>& B,
                   complex64 beta,
                   thrust::device_vector<GPU_TYPE(complex64)>& C) {
    RAW_GPU_TYPE(complex64) alpha_gpu = raw_gpu_type_cast<complex64>(alpha);
    RAW_GPU_TYPE(complex64) beta_gpu  = raw_gpu_type_cast<complex64>(beta);
    CUBLAS_CALL(cublasCgemv(handle.cublas,
                            CUBLAS_OP_N,
                            std::get<0>(A.shape),
                            std::get<1>(A.shape),
                            &alpha_gpu,
                            A.Data(),
                            std::get<1>(A.shape),
                            raw_gpu_type_cast<const complex64 *>(B.data()),
                            1,
                            &beta_gpu,
                            raw_gpu_type_cast<complex64 *>(C.data()),
                            1));
  }
};


template<template <typename> class matrix_type_gpu>
struct gemv_impl_gpu<complex128, matrix_type_gpu> {
  static void func(handle_gpu& handle,
                   complex128 alpha,
                   const matrix_type_gpu<complex128>& A,
                   const thrust::device_vector<GPU_TYPE(complex128)>& B,
                   complex128 beta,
                   thrust::device_vector<GPU_TYPE(complex128)>& C) {
    RAW_GPU_TYPE(complex128) alpha_gpu = raw_gpu_type_cast<complex128>(alpha);
    RAW_GPU_TYPE(complex128) beta_gpu  = raw_gpu_type_cast<complex128>(beta);
    CUBLAS_CALL(cublasZgemv(handle.cublas,
                            CUBLAS_OP_N,
                            std::get<0>(A.shape),
                            std::get<1>(A.shape),
                            &alpha_gpu,
                            A.Data(),
                            std::get<1>(A.shape),
                            raw_gpu_type_cast<const complex128 *>(B.data()),
                            1,
                            &beta_gpu,
                            raw_gpu_type_cast<complex128 *>(C.data()),
                            1));
  }
};


template<>
struct gemv_impl_gpu<complex64, csr_matrix_gpu> {
  static void func(handle_gpu& handle,
                   complex64 alpha,
                   const csr_matrix_gpu<complex64>& A,
                   const thrust::device_vector<GPU_TYPE(complex64)>& B,
                   complex64 beta,
                   thrust::device_vector<GPU_TYPE(complex64)>& C) {
    RAW_GPU_TYPE(complex64) alpha_gpu = raw_gpu_type_cast<complex64>(alpha);
    RAW_GPU_TYPE(complex64) beta_gpu  = raw_gpu_type_cast<complex64>(beta);
    CUSPARSE_CALL(cusparseCcsrmv(handle.cusparse,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 std::get<0>(A.shape),
                                 std::get<1>(A.shape),
                                 A.nnz,
                                 &alpha_gpu,
                                 handle.mat_descr,
                                 raw_gpu_type_cast<const complex64 *>(A.data.data()),
                                 raw_gpu_type_cast<const int *>(A.indptr.data()),
                                 raw_gpu_type_cast<const int *>(A.indices.data()),
                                 raw_gpu_type_cast<const complex64 *>(B.data()),
                                 &beta_gpu,
                                 raw_gpu_type_cast<complex64 *>(C.data())));
  }
};


template<>
struct gemv_impl_gpu<complex128, csr_matrix_gpu> {
  static void func(handle_gpu& handle,
                   complex128 alpha,
                   const csr_matrix_gpu<complex128>& A,
                   const thrust::device_vector<GPU_TYPE(complex128)>& B,
                   complex128 beta,
                   thrust::device_vector<GPU_TYPE(complex128)>& C) {
    RAW_GPU_TYPE(complex128) alpha_gpu = raw_gpu_type_cast<complex128>(alpha);
    RAW_GPU_TYPE(complex128) beta_gpu  = raw_gpu_type_cast<complex128>(beta);
    CUSPARSE_CALL(cusparseZcsrmv(handle.cusparse,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 std::get<0>(A.shape),
                                 std::get<1>(A.shape),
                                 A.nnz,
                                 &alpha_gpu,
                                 handle.mat_descr,
                                 raw_gpu_type_cast<const complex128 *>(A.data.data()),
                                 raw_gpu_type_cast<const int *>(A.indptr.data()),
                                 raw_gpu_type_cast<const int *>(A.indices.data()),
                                 raw_gpu_type_cast<const complex128 *>(B.data()),
                                 &beta_gpu,
                                 raw_gpu_type_cast<complex128 *>(C.data())));
  }
};


// general matrix-matrix multiplication
template<typename T,
         template <typename> class matrix_type_gpu_a,
         template <typename> class matrix_type_gpu_b,
         template <typename> class matrix_type_gpu_c>
struct gemm_impl_gpu {
  static void func(handle_gpu& handle,
                   T alpha,
                   const matrix_type_gpu_a<T>& A,
                   const matrix_type_gpu_b<T>& B,
                   T beta,
                   matrix_type_gpu_c<T>& C);
};

template<typename T,
         template <typename> class matrix_type_gpu_a,
         template <typename> class matrix_type_gpu_b,
         template <typename> class matrix_type_gpu_c>
void gemm_gpu(handle_gpu& handle,
             T alpha,aff
             const matrix_type_gpu_a<T>& A,
             const matrix_type_gpu_b<T>& B,
             T beta,
             matrix_type_gpu_c<T>& C) {
  gemm_impl_gpu<T, matrix_type_gpu_a, matrix_type_gpu_b, matrix_type_gpu_c>::func(
      handle, alpha, A, B, beta, C);
}


template<template <typename> class matrix_type_gpu_a,
         template <typename> class matrix_type_gpu_b,
         template <typename> class matrix_type_gpu_c>
struct gemm_impl_gpu<complex64, matrix_type_gpu_a, matrix_type_gpu_b, matrix_type_gpu_c> {
  static void func(handle_gpu& handle,aff
                   complex64 alpha,
                   const matrix_type_gpu_a<complex64>& A,
                   const matrix_type_gpu_b<complex64>& B,
                   complex64 beta,
                   matrix_type_gpu_c<complex64>& C) {
    RAW_GPU_TYPE(complex64) alpha_gpu = raw_gpu_type_cast<complex64>(alpha);
    RAW_GPU_TYPE(complex64) beta_gpu  = raw_gpu_type_cast<complex64>(beta);
    CUBLAS_CALL(cublasCgemm(handle.cublas,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            std::get<0>(A.shape),
                            std::get<1>(B.shape),
                            std::get<0>(B.shape),
                            &alpha_gpu,
                            A.Data(),
                            std::get<0>(A.shape),
                            B.Data(),
                            std::get<0>(B.shape),
                            &beta_gpu,
                            C.Data(),
                            std::get<0>(C.shape)));
  }
};


template<template <typename> class matrix_type_gpu_a,
         template <typename> class matrix_type_gpu_b,
         template <typename> class matrix_type_gpu_c>
struct gemm_impl_gpu<complex128, matrix_type_gpu_a, matrix_type_gpu_b, matrix_type_gpu_c> {
  static void func(handle_gpu& handle,
                   complex128 alpha,
                   const matrix_type_gpu_a<complex128>& A,
                   const matrix_type_gpu_b<complex128>& B,
                   complex128 beta,
                   matrix_type_gpu_c<complex128>& C) {
    RAW_GPU_TYPE(complex128) alpha_gpu = raw_gpu_type_cast<complex128>(alpha);
    RAW_GPU_TYPE(complex128) beta_gpu  = raw_gpu_type_cast<complex128>(beta);
    CUBLAS_CALL(cublasZgemm(handle.cublas,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            std::get<0>(A.shape),
                            std::get<1>(B.shape),
                            std::get<0>(B.shape),
                            &alpha_gpu,
                            A.Data(),
                            std::get<0>(A.shape),
                            B.Data(),
                            std::get<0>(B.shape),
                            &beta_gpu,
                            C.Data(),
                            std::get<0>(C.shape)));
  }
};


template<template <typename> class matrix_type_gpu_b,
         template <typename> class matrix_type_gpu_c>
struct gemm_impl_gpu<complex64, csr_matrix_gpu, matrix_type_gpu_b, matrix_type_gpu_c> {
  static void func(handle_gpu& handle,
                   complex64 alpha,
                   const csr_matrix_gpu<complex64>& A,
                   const matrix_type_gpu_b<complex64>& B,
                   complex64 beta,
                   matrix_type_gpu_c<complex64>& C) {
    RAW_GPU_TYPE(complex64) alpha_gpu = raw_gpu_type_cast<complex64>(alpha);
    RAW_GPU_TYPE(complex64) beta_gpu  = raw_gpu_type_cast<complex64>(beta);
    CUSPARSE_CALL(cusparseCcsrmm2(handle.cusparse,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  std::get<0>(A.shape),
                                  std::get<0>(C.shape),
                                  std::get<1>(A.shape),
                                  A.nnz,
                                  &alpha_gpu,
                                  handle.mat_descr,
                                  raw_gpu_type_cast<const complex64 *>(A.data.data()),
                                  raw_gpu_type_cast<const int *>(A.indptr.data()),
                                  raw_gpu_type_cast<const int *>(A.indices.data()),
                                  B.Data(),
                                  std::get<1>(B.shape),
                                  &beta_gpu,
                                  raw_gpu_type_cast<complex64 *>(C.Data()),
                                  std::get<0>(C.shape)));
  }
};


template<template <typename> class matrix_type_gpu_b,
         template <typename> class matrix_type_gpu_c>
struct gemm_impl_gpu<complex128, csr_matrix_gpu, matrix_type_gpu_b, matrix_type_gpu_c> {
  static void func(handle_gpu& handle,
                   complex128 alpha,
                   const csr_matrix_gpu<complex128>& A,
                   const matrix_type_gpu_b<complex128>& B,
                   complex128 beta,
                   matrix_type_gpu_c<complex128>& C) {
    RAW_GPU_TYPE(complex128) alpha_gpu = raw_gpu_type_cast<complex128>(alpha);
    RAW_GPU_TYPE(complex128) beta_gpu  = raw_gpu_type_cast<complex128>(beta);
    RAW_GPU_TYPE(complex128) zero      = raw_gpu_type_cast<complex128>(Zero<complex128>());
    RAW_GPU_TYPE(complex128) one       = raw_gpu_type_cast<complex128>(One<complex128>());
    CUSPARSE_CALL(cusparseZcsrmm2(handle.cusparse,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  std::get<0>(A.shape),
                                  std::get<0>(C.shape),
                                  std::get<1>(A.shape),
                                  A.nnz,
                                  &alpha_gpu,
                                  handle.mat_descr,
                                  raw_gpu_type_cast<const complex128 *>(A.data.data()),
                                  raw_gpu_type_cast<const int *>(A.indptr.data()),
                                  raw_gpu_type_cast<const int *>(A.indices.data()),
                                  B.Data(),
                                  std::get<1>(B.shape),
                                  &beta_gpu,
                                  raw_gpu_type_cast<complex128 *>(C.Data()),
                                  std::get<0>(C.shape)));
  }
};


template<template <typename> class matrix_type_gpu_a,
         template <typename> class matrix_type_gpu_c>
struct gemm_impl_gpu<complex64, matrix_type_gpu_a, csr_matrix_gpu, matrix_type_gpu_c> {
  static void func(handle_gpu& handle,
                   complex64 alpha,
                   const matrix_type_gpu_a<complex64>& A,
                   const csr_matrix_gpu<complex64>& B,
                   complex64 beta,
                   matrix_type_gpu_c<complex64>& C) {
    RAW_GPU_TYPE(complex64) alpha_gpu = raw_gpu_type_cast<complex64>(alpha);
    RAW_GPU_TYPE(complex64) beta_gpu  = raw_gpu_type_cast<complex64>(beta);
    RAW_GPU_TYPE(complex64) zero      = raw_gpu_type_cast<complex64>(Zero<complex64>());
    RAW_GPU_TYPE(complex64) one       = raw_gpu_type_cast<complex64>(One<complex64>());
    thrust::device_vector<GPU_TYPE(complex64)> A_T(std::get<0>(C.shape)*std::get<1>(C.shape));
    thrust::device_vector<GPU_TYPE(complex64)> C_T(std::get<0>(C.shape)*std::get<1>(C.shape));
    CUBLAS_CALL(cublasCgeam(handle.cublas,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            std::get<0>(A.shape),
                            std::get<1>(A.shape),
                            &one,
                            raw_gpu_type_cast<const complex64*>(A.Data()),
                            std::get<1>(A.shape),
                            &zero,
                            raw_gpu_type_cast<const complex64*>(A_T.data()),
                            std::get<0>(A.shape),
                            raw_gpu_type_cast<complex64*>(A_T.data()),
                            std::get<0>(A.shape)));
    CUBLAS_CALL(cublasCgeam(handle.cublas,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            std::get<0>(C.shape),
                            std::get<1>(C.shape),
                            &one,
                            raw_gpu_type_cast<const complex64*>(C.Data()),
                            std::get<1>(C.shape),
                            &zero,
                            raw_gpu_type_cast<const complex64*>(C_T.data()),
                            std::get<0>(C.shape),
                            raw_gpu_type_cast<complex64*>(C_T.data()),
                            std::get<0>(C.shape)));
    CUSPARSE_CALL(cusparseCcsrmm2(handle.cusparse,
                                  CUSPARSE_OPERATION_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  std::get<1>(B.shape),
                                  std::get<0>(C.shape),
                                  std::get<0>(B.shape),
                                  B.nnz,
                                  &alpha_gpu,
                                  handle.mat_descr,
                                  raw_gpu_type_cast<const complex64 *>(B.data.data()),
                                  raw_gpu_type_cast<const int *>(B.indptr.data()),
                                  raw_gpu_type_cast<const int *>(B.indices.data()),
                                  raw_gpu_type_cast<const complex64 *>(A_T.data()),
                                  std::get<1>(A.shape),
                                  &beta_gpu,
                                  raw_gpu_type_cast<complex64 *>(C_T.data()),
                                  std::get<0>(C.shape)));
    CUBLAS_CALL(cublasCgeam(handle.cublas,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            std::get<1>(C.shape),
                            std::get<0>(C.shape),
                            &one,
                            raw_gpu_type_cast<const complex64*>(C_T.data()),
                            std::get<0>(C.shape),
                            &zero,
                            raw_gpu_type_cast<const complex64*>(C.Data()),
                            std::get<1>(C.shape),
                            raw_gpu_type_cast<complex64 *>(C.Data()),
                            std::get<1>(C.shape)));
  }
};


template<template <typename> class matrix_type_gpu_a,
         template <typename> class matrix_type_gpu_c>
struct gemm_impl_gpu<complex128, matrix_type_gpu_a, csr_matrix_gpu, matrix_type_gpu_c> {
  static void func(handle_gpu& handle,
                   complex128 alpha,
                   const matrix_type_gpu_a<complex128>& A,
                   const csr_matrix_gpu<complex128>& B,
                   complex128 beta,
                   matrix_type_gpu_c<complex128>& C) {
    RAW_GPU_TYPE(complex128) alpha_gpu = raw_gpu_type_cast<complex128>(alpha);
    RAW_GPU_TYPE(complex128) beta_gpu  = raw_gpu_type_cast<complex128>(beta);
    RAW_GPU_TYPE(complex128) zero      = raw_gpu_type_cast<complex128>(Zero<complex128>());
    RAW_GPU_TYPE(complex128) one       = raw_gpu_type_cast<complex128>(One<complex128>());
    thrust::device_vector<GPU_TYPE(complex128)> A_T(std::get<0>(C.shape)*std::get<1>(C.shape));
    thrust::device_vector<GPU_TYPE(complex128)> C_T(std::get<0>(C.shape)*std::get<1>(C.shape));
    CUBLAS_CALL(cublasZgeam(handle.cublas,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            std::get<0>(A.shape),
                            std::get<1>(A.shape),
                            &one,
                            raw_gpu_type_cast<const complex128*>(A.Data()),
                            std::get<1>(A.shape),
                            &zero,
                            raw_gpu_type_cast<const complex128*>(A_T.data()),
                            std::get<0>(A.shape),
                            raw_gpu_type_cast<complex128*>(A_T.data()),
                            std::get<0>(A.shape)));
    CUBLAS_CALL(cublasZgeam(handle.cublas,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            std::get<0>(C.shape),
                            std::get<1>(C.shape),
                            &one,
                            raw_gpu_type_cast<const complex128*>(C.Data()),
                            std::get<1>(C.shape),
                            &zero,
                            raw_gpu_type_cast<const complex128*>(C_T.data()),
                            std::get<0>(C.shape),
                            raw_gpu_type_cast<complex128*>(C_T.data()),
                            std::get<0>(C.shape)));
    CUSPARSE_CALL(cusparseZcsrmm2(handle.cusparse,
                                  CUSPARSE_OPERATION_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  std::get<1>(B.shape),
                                  std::get<0>(C.shape),
                                  std::get<0>(B.shape),
                                  B.nnz,
                                  &alpha_gpu,
                                  handle.mat_descr,
                                  raw_gpu_type_cast<const complex128 *>(B.data.data()),
                                  raw_gpu_type_cast<const int *>(B.indptr.data()),
                                  raw_gpu_type_cast<const int *>(B.indices.data()),
                                  raw_gpu_type_cast<const complex128 *>(A_T.data()),
                                  std::get<1>(A.shape),
                                  &beta_gpu,
                                  raw_gpu_type_cast<complex128 *>(C_T.data()),
                                  std::get<0>(C.shape)));
    CUBLAS_CALL(cublasZgeam(handle.cublas,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            std::get<1>(C.shape),
                            std::get<0>(C.shape),
                            &one,
                            raw_gpu_type_cast<const complex128*>(C_T.data()),
                            std::get<0>(C.shape),
                            &zero,
                            raw_gpu_type_cast<const complex128*>(C.Data()),
                            std::get<1>(C.shape),
                            raw_gpu_type_cast<complex128 *>(C.Data()),
                            std::get<1>(C.shape)));
  }
};


}

#endif  /* CUBLAS_CUSPARSE_WRAPPER_H */
