/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/
#include "gpu_info.h"

#include <cuda.h>

#include "utility_gpu.h"

namespace libheom {

bool gpu_init_flag = false;

void init_cuda_driver() {
  if (not gpu_init_flag) {
    if (cuInit(0) != CUDA_SUCCESS) {
      std::cerr << "[Error] Initialization of CUDA Runtime driver failed." << std::endl;
      std::exit(1);
    }
    gpu_init_flag = true;
  }
}

int GetGpuDeviceCount() {
  int result;
  init_cuda_driver();
  cudaGetDeviceCount(&result);
  return result;
}

const std::string GetGpuDeviceName(int device_number) {
  cudaDeviceProp devprop;
  init_cuda_driver();
  CUDA_CALL(cudaGetDeviceProperties(&devprop, device_number));
  return std::string(devprop.name);
}

void SetGpuDevice(int selected)
{
  int num_dev = GetGpuDeviceCount();
  if (num_dev == 0) {
    std::cerr << "[Error] NVIDIA GPU unit is not found." << std::endl;
    std::exit(1);
  }
  if (0 > selected || selected >= num_dev) {
    std::cerr << "[Error] " << selected << " is out of range of valid device number" << std::endl;
    std::exit(1);
  }
  
  cudaSetDevice(selected);
  cudaDeviceProp devprop;
  CUDA_CALL(cudaGetDeviceProperties(&devprop, selected));
};

}