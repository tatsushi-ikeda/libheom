# -*- mode:cmake -*-
#  LibHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LINCENSE.txt for licence.
# ------------------------------------------------------------------------*/

set(NONHEADER_SOURCE    hrchy_space.cc utility_mkl.cc utility.cc)
set(NONHEADER_SOURCE_CU linalg_engine_cuda.cu utility_cuda.cu)
set(LIBRARIES_CU        cuda cudart cublas cusparse cusolver)

if (ENABLE_CUDA)
  add_library(libheom_core STATIC ${NONHEADER_SOURCE} ${NONHEADER_SOURCE_CU})
  target_link_libraries(libheom_core PRIVATE "${LIBRARIES_CU}")
  set_target_properties(libheom_core PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
else()
  add_library(libheom_core STATIC ${NONHEADER_SOURCE})
endif()

target_include_directories(libheom_core
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)
add_dependencies(libheom_core generated_headers)

if (ENABLE_MKL AND BLAS_FOUND)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
  target_link_libraries(libheom_core PRIVATE libmkl_rt.so "${BLAS_LIBRARIES}")
endif()

# set(PYBIND11_PYTHON_VERSION "3.6")

# if (ENABLE_CUDA)
#   process_jinja2(entrypoint.cu SOURCE entrypoint.cc.j2)
#   pybind11_add_module(libheom entrypoint.cu)
#   set_target_properties(libheom PROPERTIES LINKER_LANGUAGE CUDA)
#   set_target_properties(libheom PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
# else()
#   process_jinja2(entrypoint.cc)
#   pybind11_add_module(libheom entrypoint.cc)
#   target_link_libraries(libheom PRIVATE libheom_core)
# endif()
# target_link_libraries(libheom PRIVATE libheom_core)
