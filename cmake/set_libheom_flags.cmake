# -*- mode: cmake -*-
#  LibHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LINCENSE.txt for licence.
# ------------------------------------------------------------------------*/

if(NOT STACKTRACE)
  set(LIBHEOM_FLAGS "${LIBHEOM_FLAGS} -DQL_NDEBUG")
endif()

if(ENABLE_SINGLE)
  set(LIBHEOM_FLAGS "${LIBHEOM_FLAGS} -DENABLE_SINGLE")
endif()

if(ENABLE_DOUBLE)
  set(LIBHEOM_FLAGS "${LIBHEOM_FLAGS} -DENABLE_DOUBLE")
endif()

if(ASSUME_HERMITIAN)
  set(LIBHEOM_FLAGS "${LIBHEOM_FLAGS} -DASSUME_HERMITIAN")
endif()

if(ENABLE_EIGEN)
  set(LIBHEOM_FLAGS "${LIBHEOM_FLAGS} -DENABLE_EIGEN")
endif()

if(ENABLE_MKL)
  set(LIBHEOM_FLAGS "${LIBHEOM_FLAGS} -DENABLE_MKL")
endif()

if(ENABLE_CUDA)
  set(LIBHEOM_FLAGS "${LIBHEOM_FLAGS} -DENABLE_CUDA")
endif()  
