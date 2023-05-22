/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifdef ENABLE_MKL
#include "linalg_engine/utility_mkl.h"

namespace libheom {

// The source of these messages is https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/inspector-executor-sparse-blas-routines/matrix-manipulation-routines/mkl-sparse-create-csr.html
std::map<sparse_status_t, std::string> MKL_SPARSE_ERR_MSG = {{
  { SPARSE_STATUS_SUCCESS,
    "SPARSE_STATUS_SUCCESS: The operation was successful." },
  { SPARSE_STATUS_NOT_INITIALIZED,
    "SPARSE_STATUS_NOT_INITIALIZED: The routine encountered an empty handle or matrix array." },
  { SPARSE_STATUS_ALLOC_FAILED,
    "SPARSE_STATUS_ALLOC_FAILED: Internal memory allocation failed." },
  { SPARSE_STATUS_INVALID_VALUE,
    "SPARSE_STATUS_INVALID_VALUE: The input parameters contain an invalid value." },
  { SPARSE_STATUS_EXECUTION_FAILED,
    "SPARSE_STATUS_EXECUTION_FAILED: Execution failed." },
  { SPARSE_STATUS_INTERNAL_ERROR,
    "SPARSE_STATUS_INTERNAL_ERROR: An error in algorithm implementation occurred." },
  { SPARSE_STATUS_NOT_SUPPORTED,
    "SPARSE_STATUS_NOT_SUPPORTED: The requested operation is not supported." }, }};

} // namespace libheom
#endif // ifdef ENABLE_MKL
