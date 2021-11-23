/*
 * LibHEOM: Copyright (c) Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/
#include <cuda.h>

#include "gpu_info.h"
#include "utility_gpu.h"

namespace libheom
{


bool gpu_init_flag = false;


void init_cuda_driver
/**/()
{
  if (not gpu_init_flag) {
    if (cuInit(0) != CUDA_SUCCESS) {
      std::cerr << "[Error] Initialization of CUDA Runtime driver failed." << std::endl;
      std::exit(1);
    }
    gpu_init_flag = true;
  }
}


int get_gpu_device_count
/**/()
{
  int result;
  init_cuda_driver();
  cudaGetDeviceCount(&result);
  return result;
}


const std::string get_gpu_device_name
/**/(int device_number)
{
  cudaDeviceProp devprop;
  init_cuda_driver();
  CUDA_CALL(cudaGetDeviceProperties(&devprop, device_number));
  return std::string(devprop.name);
}


void set_gpu_device
/**/(int selected)
{
  int num_dev = get_gpu_device_count();
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