/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef UTILITY_GPU_H
#define UTILITY_GPU_H
 
#include <iostream>
#include <string>
#include <map>
#include <cublas_v2.h>
#include <cusparse.h>

namespace libheom {
#define CUDA_CALL(func)                                                    \
  {                                                                        \
    cudaError_t err = (func);                                              \
    if (err != cudaSuccess) {                                              \
      std::cerr << "[Error] "                                              \
                << cudaGetErrorString(err) << " "                          \
                << "(error code: " << err << ") "                          \
                << "at " << __FILE__ << " line " << __LINE__ << std::endl; \
      std::exit(1);                                                        \
    }                                                                      \
  }

#define CUDA_CHECK_KERNEL_CALL()                                           \
  {                                                                        \
    cudaError_t err = cudaGetLastError();                                  \
    if (err != cudaSuccess) {                                              \
      std::cerr << "[Error] "                                              \
                << cudaGetErrorString(err) << " "                          \
                << "(error code: " << err << ") "                          \
                << "at " << __FILE__ << " line " << __LINE__ << std::endl; \
      std::exit(1);                                                        \
    }                                                                      \
  }

extern std::map<cublasStatus_t,std::string> CUBLAS_ERR_MSG;
#define CUBLAS_CALL(func)                                                  \
  {                                                                        \
    cublasStatus_t err = (func);                                           \
    if (err != CUBLAS_STATUS_SUCCESS) {                                    \
      std::cerr << "[Error:cuBLAS]  "                                   \
                << "(error code: " << err << ") "                          \
                << "at " << __FILE__ << " line " << __LINE__ << std::endl; \
      std::cerr << CUBLAS_ERR_MSG[err] << std::endl;                    \
      std::exit(1);                                                        \
    }                                                                      \
  }

extern std::map<cusparseStatus_t,std::string> CUSPARSE_ERR_MSG;
#define CUSPARSE_CALL(func)                                                \
  {                                                                        \
    cusparseStatus_t err = (func);                                         \
    if (err != CUSPARSE_STATUS_SUCCESS) {                                  \
      std::cerr << "[Error:cuSPARSE] "                                  \
                << "(error code: " << err << ") "                          \
                << "at " << __FILE__ << " line " << __LINE__ << std::endl; \
      std::cerr << CUSPARSE_ERR_MSG[err] << std::endl;                     \
      std::exit(1);                                                        \
    }                                                                      \
  }

}


#endif
