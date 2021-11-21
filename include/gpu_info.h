/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef GPU_INFO_H
#define GPU_INFO_H

#include <string>

namespace libheom {

int get_gpu_device_count();
const std::string get_gpu_device_name(int device_number);
void set_gpu_device(int selected);

}

#endif /* GPU_INFO_H */
