/*
 * LibHEOM: Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#include "handle_gpu.h"

#include "gpu_info.h"

namespace libheom {

void handle_gpu::init(int device_number) {
  this->device_number = device_number;
  SetGpuDevice(device_number);
  
  CUSPARSE_CALL(cusparseCreate(&cusparse));
  CUBLAS_CALL(cublasCreate(&cublas));
  
  CUSPARSE_CALL(cusparseCreateMatDescr(&mat_descr));
  CUSPARSE_CALL(cusparseSetMatType(mat_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_CALL(cusparseSetMatIndexBase(mat_descr, CUSPARSE_INDEX_BASE_ZERO));
}

}