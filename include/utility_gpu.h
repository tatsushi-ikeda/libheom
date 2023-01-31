/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef LIBHEOM_UTILITY_GPU_H
#define LIBHEOM_UTILITY_GPU_H

namespace libheom
{

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

}

#endif
