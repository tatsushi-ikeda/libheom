/*
 * LibHEOM: Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#include "utility_gpu.h"

namespace libheom {

// The source of these messages is https://docs.nvidia.com/cuda/cublas/index.html.
std::map<cublasStatus_t,std::string> CUBLAS_ERR_MSG =
  {{
      {CUBLAS_STATUS_SUCCESS,
       "CUBLAS_STATUS_SUCCESS: The operation completed successfully."},
      {CUBLAS_STATUS_NOT_INITIALIZED,
       "CUBLAS_STATUS_NOT_INITIALIZED: The cuBLAS library was not initialized. This is usually caused by the lack of a prior cublasCreate() call, an error in the CUDA Runtime API called by the cuBLAS routine, or an error in the hardware setup."},
      {CUBLAS_STATUS_ALLOC_FAILED,
       "CUBLAS_STATUS_ALLOC_FAILED: Resource allocation failed inside the cuBLAS library. This is usually caused by a cudaMalloc() failure."},
      {CUBLAS_STATUS_INVALID_VALUE,
       "CUBLAS_STATUS_INVALID_VALUE: An unsupported value or parameter was passed to the function (a negative vector size, for example)."},
      {CUBLAS_STATUS_ARCH_MISMATCH,
       "CUBLAS_STATUS_ARCH_MISMATCH: The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision."},
      {CUBLAS_STATUS_MAPPING_ERROR,
       "CUBLAS_STATUS_MAPPING_ERROR: An access to GPU memory space failed, which is usually caused by a failure to bind a texture."},
      {CUBLAS_STATUS_EXECUTION_FAILED,
       "CUBLAS_STATUS_EXECUTION_FAILED: The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons."},
      {CUBLAS_STATUS_INTERNAL_ERROR,
       "CUBLAS_STATUS_INTERNAL_ERROR: An internal cuBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure."},
      {CUBLAS_STATUS_NOT_SUPPORTED,
       "CUBLAS_STATUS_NOT_SUPPORTED: The functionnality requested is not supported."},
      {CUBLAS_STATUS_LICENSE_ERROR,
       "CUBLAS_STATUS_LICENSE_ERROR: The functionnality requested requires some license and an error was detected when trying to check the current licensing. This error can happen if the license is not present or is expired or if the environment variable NVIDIA_LICENSE_FILE is not set properly."}
    }};


// The source of these messages is from https://docs.nvidia.com/cuda/cusparse/index.html.
std::map<cusparseStatus_t,std::string> CUSPARSE_ERR_MSG =
  {{
      {CUSPARSE_STATUS_SUCCESS,
       "CUSPARSE_STATUS_SUCCESS: The operation completed successfully."},
      {CUSPARSE_STATUS_NOT_INITIALIZED,
       "CUSPARSE_STATUS_NOT_INITIALIZED: The cuSPARSE library was not initialized. This is usually caused by the lack of a prior call, an error in the CUDA Runtime API called by the cuSPARSE routine, or an error in the hardware setup."},
      {CUSPARSE_STATUS_ALLOC_FAILED,
       "CUSPARSE_STATUS_ALLOC_FAILED: Resource allocation failed inside the cuSPARSE library. This is usually caused by a cudaMalloc() failure."},
      {CUSPARSE_STATUS_INVALID_VALUE,
       "CUSPARSE_STATUS_INVALID_VALUE: An unsupported value or parameter was passed to the function (a negative vector size, for example)."},
      {CUSPARSE_STATUS_ARCH_MISMATCH,
       "CUSPARSE_STATUS_ARCH_MISMATCH: The function requires a feature absent from the device architecture; usually caused by the lack of support for atomic operations or double precision."},
      {CUSPARSE_STATUS_MAPPING_ERROR,
       "CUSPARSE_STATUS_MAPPING_ERROR: An access to GPU memory space failed, which is usually caused by a failure to bind a texture."},
      {CUSPARSE_STATUS_EXECUTION_FAILED,
       "CUSPARSE_STATUS_EXECUTION_FAILED: The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons."},
      {CUSPARSE_STATUS_INTERNAL_ERROR,
       "CUSPARSE_STATUS_INTERNAL_ERROR: An internal cuSPARSE operation failed. This error is usually caused by a cudaMemcpyAsync() failure."},
      {CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED,
       "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: The matrix type is not supported by this function. This is usually caused by passing an invalid matrix descriptor to the function."}
    }};

}