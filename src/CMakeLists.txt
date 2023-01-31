# -*- mode:cmake -*-
#  LibHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LINCENSE.txt for licence.
# ------------------------------------------------------------------------*/

set(LIBHEOM_CXX_SOURCE  hrchy_space.cc utility_mkl.cc utility.cc)
set(LIBHEOM_CUDA_SOURCE linalg_engine_cuda.cu utility_cuda.cu)

if (LIBHEOM_ENABLE_CUDA_INTERNAL)
  add_library(libheom STATIC ${LIBHEOM_CXX_SOURCE} ${LIBHEOM_CUDA_SOURCE})
  set_target_properties(libheom PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
else()
  add_library(libheom STATIC ${LIBHEOM_CXX_SOURCE})
endif()
target_link_libraries(libheom PRIVATE ${LIBHEOM_LIBRARIES})

target_include_directories(libheom
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)
add_dependencies(libheom generated_headers)
