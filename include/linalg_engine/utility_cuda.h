/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef UTILITY_CUDA_H
#define UTILITY_CUDA_H

#include <iostream>
#include <string>
#include <map>
#include "linalg_engine/include_cuda.h"

namespace libheom
{

extern std::map<cublasStatus_t,std::string> CUBLAS_ERR_MSG;
#define CUBLAS_CALL(func)                                                  \
  {                                                                        \
    cublasStatus_t err = (func);                                           \
    if (err != CUBLAS_STATUS_SUCCESS) {                                    \
      std::cerr << "[Error:cuBLAS]  "                                      \
                << "(error code: " << err << ") "                          \
                << "at " << __FILE__ << " line " << __LINE__ << std::endl; \
      std::cerr << CUBLAS_ERR_MSG[err] << std::endl;                       \
      std::exit(1);                                                        \
    }                                                                      \
  }

extern std::map<cusparseStatus_t,std::string> CUSPARSE_ERR_MSG;
#define CUSPARSE_CALL(func)                                                \
  {                                                                        \
    cusparseStatus_t err = (func);                                         \
    if (err != CUSPARSE_STATUS_SUCCESS) {                                  \
      std::cerr << "[Error:cuSPARSE] "                                     \
                << "(error code: " << err << ") "                          \
                << "at " << __FILE__ << " line " << __LINE__ << std::endl; \
      std::cerr << CUSPARSE_ERR_MSG[err] << std::endl;                     \
      std::exit(1);                                                        \
    }                                                                      \
  }

extern std::map<cusolverStatus_t,std::string> CUSOLVER_ERR_MSG;
#define CUSOLVER_CALL(func)                                                \
  {                                                                        \
    cusolverStatus_t err = (func);                                         \
    if (err != CUSOLVER_STATUS_SUCCESS) {                                  \
      std::cerr << "[Error:cuSOLVER] "                                     \
                << "(error code: " << err << ") "                          \
                << "at " << __FILE__ << " line " << __LINE__ << std::endl; \
      std::cerr << CUSOLVER_ERR_MSG[err] << std::endl;                     \
      std::exit(1);                                                        \
    }                                                                      \
  }

}

#endif
