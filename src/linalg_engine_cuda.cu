/* -*- mode:cuda -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#include "linalg_engine/linalg_engine_cuda.h"

// https://qiita.com/gyu-don/items/ef8a128fa24f6bddd342

namespace libheom {

constexpr unsigned int CUDA_BLOCK_SIZE = 1024;

// template<typename dtype_real>
// __global__ void errnorm1_kernel(device_t<complex_t<dtype_real>,env_gpu>* e,
//                                 device_t<complex_t<dtype_real>,env_gpu>* x,
//                                 dtype_real atol,
//                                 dtype_real rtol,
//                                 int n,
//                                 device_t<dtype_real,env_gpu>* work)
// {
//   // return e[i]/(atol + rtol*x[i]);
//   extern __shared__ dtype_real smem[];

//   unsigned int tid = threadIdx.x;
//   unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
//   dtype_real result = (i < n) ? cuCabsf(e[i])/(atol + rtol*cuCabsf(x[i])) : 0;
//   if (i + blockDim.x < n) {
//     result += cuCabsf(e[i + blockDim.x])/(atol + rtol*cuCabsf(x[i + blockDim.x]));
//   }
//   smem[tid] = result;
//   __syncthreads();

//   for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
//     if (tid < s) {
//       smem[tid] = result = result + smem[tid + s];
//     }
//     __syncthreads();
//   }
//   if (tid < 32) {
//     if(blockDim.x >= 64) result += smem[tid + 32];
//     for (int offset = 32/2; offset>0; offset>>=1) {
//       result += __shfl_down(result, offset);
//     }
//   }
//   if (tid == 0) {
//     work[blockIdx.x] = result;
//   }
// }


__global__ void errnorm1_c_kernel(device_t<complex_t<float32>, env_gpu> *e,
                                  device_t<complex_t<float32>, env_gpu> *x_1,
                                  device_t<complex_t<float32>, env_gpu> *x_2,
                                  float32                                atol,
                                  float32                                rtol,
                                  int                                    n,
                                  device_t<float32, env_gpu>            *work)
{
  // return e[i]/(atol + rtol*x[i]);
  extern __shared__ float32 smem32[];

  unsigned int tid = threadIdx.x;
  unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  float32      x_val, elem;
  float32      result = 0.0f;
  if (i < n) {
    x_val  = (cuCabsf(x_1[i]) > cuCabsf(x_2[i])) ? cuCabsf(x_1[i]) : cuCabsf(x_2[i]);
    elem   = cuCabsf(e[i]) / (atol + rtol * x_val);
    result = elem * elem;
  }
  if (i + blockDim.x < n) {
    x_val   =
      (cuCabsf(x_1[i + blockDim.x]) >
       cuCabsf(x_2[i + blockDim.x])) ? cuCabsf(x_1[i + blockDim.x]) : cuCabsf(x_2[i + blockDim.x]);
    elem    = cuCabsf(e[i + blockDim.x]) / (atol + rtol * x_val);
    result += elem * elem;
  }
  smem32[tid] = result;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      smem32[tid] = result = result + smem32[tid + s];
    }
    __syncthreads();
  }
  if (tid < 32) {
    if (blockDim.x >= 64) { result += smem32[tid + 32]; }
    for (int offset = 32 / 2; offset > 0; offset >>= 1) {
      result += __shfl_down_sync(0xffffffff, result, offset);
      // result += __shfl_down(result, offset);
    }
  }
  if (tid == 0) {
    work[blockIdx.x] = result;
  }
}

__global__ void errnorm1_z_kernel(device_t<complex_t<float64>, env_gpu> *e,
                                  device_t<complex_t<float64>, env_gpu> *x_1,
                                  device_t<complex_t<float64>, env_gpu> *x_2,
                                  float64                                atol,
                                  float64                                rtol,
                                  int                                    n,
                                  device_t<float64, env_gpu>            *work)
{
  // return e[i]/(atol + rtol*x[i]);
  extern __shared__ float64 smem64[];

  unsigned int tid = threadIdx.x;
  unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  float64      x_val, elem;
  float64      result = 0.0;
  if (i < n) {
    x_val  = (cuCabs(x_1[i]) > cuCabs(x_2[i])) ? cuCabs(x_1[i]) : cuCabs(x_2[i]);
    elem   = cuCabs(e[i]) / (atol + rtol * x_val);
    result = elem * elem;
  }
  if (i + blockDim.x < n) {
    x_val   =
      (cuCabs(x_1[i + blockDim.x]) >
       cuCabs(x_2[i + blockDim.x])) ? cuCabs(x_1[i + blockDim.x]) : cuCabs(x_2[i + blockDim.x]);
    elem    = cuCabs(e[i + blockDim.x]) / (atol + rtol * x_val);
    result += elem * elem;
  }
  smem64[tid] = result;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      smem64[tid] = result = result + smem64[tid + s];
    }
    __syncthreads();
  }
  if (tid < 32) {
    if (blockDim.x >= 64) { result += smem64[tid + 32]; }
    for (int offset = 32 / 2; offset > 0; offset >>= 1) {
      result += __shfl_down_sync(0xffffffff, result, offset);
      // result += __shfl_down(result, offset);
    }
  }
  if (tid == 0) {
    work[blockIdx.x] = result;
  }
}

float32 errnorm1_C(device_t<complex_t<float32>, env_gpu> *e,
                   device_t<complex_t<float32>, env_gpu> *x_1,
                   device_t<complex_t<float32>, env_gpu> *x_2,
                   float32                                atol,
                   float32                                rtol,
                   int                                    n_level)
{
  CALL_TRACE();
  unsigned int                grid   = n_level / CUDA_BLOCK_SIZE + 1;
  float32                     result = zero<float32>();
  float32                    *work;
  device_t<float32, env_gpu> *work_dev;
  work = new float32 [grid];
  CUDA_CALL(cudaMalloc(&work_dev, grid * sizeof(float32)));
  errnorm1_c_kernel<<<grid, CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE *sizeof(float32)>>>(e,
                                                                                 x_1,
                                                                                 x_2,
                                                                                 atol,
                                                                                 rtol,
                                                                                 n_level,
                                                                                 work_dev);
  CUDA_CHECK_KERNEL_CALL();
  CUDA_CALL(cudaMemcpy(work, work_dev, grid * sizeof(float32), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaFree(work_dev));
  for (int i = 0; i < grid; ++i) {
    result += work[i];
  }
  delete [] work;
  return std::sqrt(result / n_level);
}

float64 errnorm1_Z(device_t<complex_t<float64>, env_gpu> *e,
                   device_t<complex_t<float64>, env_gpu> *x_1,
                   device_t<complex_t<float64>, env_gpu> *x_2,
                   float64                                atol,
                   float64                                rtol,
                   int                                    n_level)
{
  CALL_TRACE();
  unsigned int                grid   = n_level / CUDA_BLOCK_SIZE + 1;
  float64                     result = zero<float64>();
  float64                    *work;
  device_t<float64, env_gpu> *work_dev;
  work = new float64 [grid];
  CUDA_CALL(cudaMalloc(&work_dev, grid * sizeof(float64)));
  errnorm1_z_kernel<<<grid, CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE *sizeof(float64)>>>(e,
                                                                                 x_1,
                                                                                 x_2,
                                                                                 atol,
                                                                                 rtol,
                                                                                 n_level,
                                                                                 work_dev);
  CUDA_CHECK_KERNEL_CALL();
  CUDA_CALL(cudaMemcpy(work, work_dev, grid * sizeof(float64), cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaFree(work_dev));
  for (int i = 0; i < grid; ++i) {
    result += work[i];
  }
  delete [] work;
  return std::sqrt(result / n_level);
}

} // namespace libheom
