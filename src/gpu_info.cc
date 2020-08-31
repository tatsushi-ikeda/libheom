/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#include "gpu_info.h"

#include <iostream>

namespace libheom {

int GetGpuDeviceCount() {
  return 0;
}

const std::string GetGpuDeviceName(int device_number) {
  std::cerr << "[Error] GPU parallerization is not supported." << std::endl;
  std::exit(1);
  return std::string("");
}

}
