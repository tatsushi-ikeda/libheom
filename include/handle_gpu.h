/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef HANDLE_GPU
#define HANDLE_GPU

#include "utility_gpu.h"

namespace libheom {

class handle_gpu {
public:
  int                device_number = 0;
  cusparseHandle_t   cusparse;
  cublasHandle_t     cublas;  
  cusparseMatDescr_t mat_descr;

  void initialize(int device_number);
};

}

#endif /* HANDLE_GPU */

