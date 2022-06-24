# -*- mode:cmake -*-
#  LibHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LINCENSE.txt for licence.
# ------------------------------------------------------------------------*/

set(GERENATED_HEADERS
  lil_matrix.h
  linalg_engine_eigen.h
  linalg_engine_nil.h
  linalg_engine_mkl.h
  linalg_engine_cuda.h
  dense_matrix_eigen.h
  sparse_matrix_eigen.h
  sparse_matrix_mkl.h)

add_custom_target(generated_headers DEPENDS ${GERENATED_HEADERS})

foreach(header IN LISTS GERENATED_HEADERS)
  process_jinja2(${header})
endforeach()
