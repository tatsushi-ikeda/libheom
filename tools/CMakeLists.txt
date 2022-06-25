# -*- mode: cmake -*-
#  LibHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LINCENSE.txt for licence.
# ------------------------------------------------------------------------*/

set(ENGINE_LIST)
if(LIBHEOM_ENABLE_EIGEN)
  list(APPEND ENGINE_LIST "\"eigen\"")
endif()
if(LIBHEOM_ENABLE_MKL)
  list(APPEND ENGINE_LIST "\"mkl\"")
endif()
if(LIBHEOM_ENABLE_CUDA)
  list(APPEND ENGINE_LIST "\"cuda\"")
endif()
string(REPLACE ";" ", " ENGINE_LIST_STR "${ENGINE_LIST}")

set(DTYPE_LIST)
if(LIBHEOM_ENABLE_SINGLE)
  list(APPEND DTYPE_LIST "[\"complex64\",\"c\"]")
endif()
if(LIBHEOM_ENABLE_DOUBLE)
  list(APPEND DTYPE_LIST "[\"complex128\",\"z\"]")
endif()
string(REPLACE ";" ", " DTYPE_LIST_STR "${DTYPE_LIST}")

configure_file(params.json.in params.json)
