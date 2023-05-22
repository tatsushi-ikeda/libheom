/* -*- mode:cuda -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef LIBHEOM_ENV_GPU_H
#define LIBHEOM_ENV_GPU_H

#include "env.h"

namespace libheom {

class env_gpu : public env_base {};

}

#ifdef ENABLE_CUDA
#include "utility_gpu.h"
#include <cublas_v2.h>

namespace libheom {

template<> struct device_type<int,       env_gpu> { typedef int value; };
template<> struct device_type<float32,   env_gpu> { typedef float value; };
template<> struct device_type<float64,   env_gpu> { typedef double value; };
template<> struct device_type<complex64, env_gpu> { typedef cuFloatComplex value; };
template<> struct device_type<complex128, env_gpu> { typedef cuDoubleComplex value; };

template<typename dtype, bool mirror>
struct new_dev_impl<dtype, env_gpu, mirror> {
  inline static device_t<dtype, env_gpu> *func(int size)
  {
    CALL_TRACE();
    device_t<dtype, env_gpu> *ptr;
    CUDA_CALL(cudaMalloc(&ptr, size * sizeof(device_t<dtype, env_gpu>)));
    return ptr;
  }
};


template<typename dtype, bool mirror>
struct delete_dev_impl<dtype, env_gpu, mirror> {
  inline static void func(device_t<dtype, env_gpu> *ptr)
  {
    CALL_TRACE();
    CUDA_CALL(cudaFree(ptr));
  }
};


template<typename dtype>
struct host2dev_impl<dtype, env_gpu> {
  inline static void func(dtype * const &ptr_host, device_t<dtype, env_gpu> * &ptr_dev, int size)
  {
    CALL_TRACE();
    CUDA_CALL(cudaMemcpy(ptr_dev, ptr_host, size * sizeof(dtype), cudaMemcpyHostToDevice));
  }
};


template<typename dtype>
struct dev2host_impl<dtype, env_gpu> {
  inline static void func(device_t<dtype, env_gpu> * &ptr_dev, dtype * const &ptr_host, int size)
  {
    CALL_TRACE();
    CUDA_CALL(cudaMemcpy(ptr_host, ptr_dev, size * sizeof(dtype), cudaMemcpyDeviceToHost));
  }
};


} // namespace libheom
#endif // ifdef ENABLE_CUDA
#endif // ifndef LIBHEOM_ENV_GPU_H
