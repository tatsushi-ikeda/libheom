#  -*- mode:cmake -*-
#  LibHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LINCENSE.txt for licence.
# ------------------------------------------------------------------------*/

cmake_minimum_required(VERSION 3.12)

file(READ "VERSION.txt" VER)
string(STRIP "${VER}" VER)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

set(ENABLE_CUDA AUTO CACHE STRING "Enable GPU parallelization support")
set_property(CACHE ENABLE_CUDA PROPERTY STRINGS AUTO ON OFF)
if (ENABLE_CUDA STREQUAL "AUTO")
  find_program(NVCC_PATH nvcc)
  if(NVCC_PATH STREQUAL "NVCC_PATH-NOTFOUND")
    set(ENABLE_CUDA OFF)
  else()
    set(ENABLE_CUDA ON)
  endif()
endif()
message(STATUS "ENABLE_CUDA: ${ENABLE_CUDA}")

if (ENABLE_CUDA)
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  set(CUDAHOSTCXX              ${CMAKE_CXX_COMPILER})
  set(CMAKE_CXX_STANDARD  17)
  set(CMAKE_CUDA_STANDARD 17)
  project(libheom LANGUAGES CXX CUDA)
else()
  set(CMAKE_CXX_STANDARD  17)
  project(libheom LANGUAGES CXX)
endif()
set(PROJECT_VERSION ${VER})

set(STACKTRACE OFF CACHE STRING "Enable call stack trace support for debug")
set_property(CACHE STACKTRACE PROPERTY STRINGS ON OFF)

include(CheckIncludeFileCXX)

set(ENABLE_EIGEN ON CACHE STRING "Enable Intel EIGEN support")
set_property(CACHE ENABLE_EIGEN PROPERTY STRINGS ON OFF)
message(STATUS "ENABLE_EIGEN: ${ENABLE_EIGEN}")

set(ENABLE_MKL AUTO CACHE STRING "Enable Intel MKL support")
set_property(CACHE ENABLE_MKL PROPERTY STRINGS AUTO ON OFF)
if (ENABLE_MKL STREQUAL "AUTO")
  CHECK_INCLUDE_FILE_CXX(mkl.h MKL_PATH)
  if(MKL_PATH)
    set(ENABLE_MKL ON)
  else()
    set(ENABLE_MKL OFF)
  endif()
endif()
message(STATUS "ENABLE_MKL: ${ENABLE_MKL}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PythonInterp 3.6 REQUIRED)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmax-errors=1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-enum-compare")
endif()

set(ENABLE_SINGLE ON CACHE STRING "Enable single point precision")
set_property(CACHE ENABLE_SINGLE PROPERTY STRINGS ON OFF)

set(ENABLE_DOUBLE ON CACHE STRING "Enable double point precision")
set_property(CACHE ENABLE_DOUBLE PROPERTY STRINGS ON OFF)

set(ASSUME_HERMITIAN OFF)
# set(ASSUME_HERMITIAN ON CACHE STRING "Assume Hermite matrices")
# set_property(CACHE ASSUME_HERMITIAN PROPERTY STRINGS ON OFF)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-fabi-version=0" COMPILER_OPT_FABI_VERSION_SUPPORTED)
if(COMPILER_OPT_FABI_VERSION_SUPPORTED)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fabi-version=0")
endif()


find_package(OpenMP)
if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_SHARED_LIBRARY_CUDA_FLAGS "${CMAKE_SHARED_LIBRARY_CUDA_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(ENABLE_EIGEN)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/eigen")
    include_directories(3rdparty/eigen)
    message(STATUS "Found Eigen Library in 3rdparty/ directory")
  endif()
endif()

if(ENABLE_MKL)
  set(BLA_STATIC OFF)
  set(BLA_VENDOR Intel10_64lp)
  find_package(BLAS)
  if (NOT BLAS_FOUND)
    message(FATAL_ERROR "Could not find MKL Library while ENABLE_MKL=ON")
  endif()
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_DIRECT_CALL")
endif()

# if(ENABLE_MKL)
#   if(ENABLE_EIGEN)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_USE_MKL_ALL")
#   endif()
# endif()

if(ENABLE_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${CMAKE_CXX_FLAGS}\"")
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler \"${CMAKE_CXX_FLAGS_RELEASE}\"")
  if(NOT CUDA_ARCH_LIST)
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS)

    string(STRIP "${INSTALLED_GPU_CCS}" INSTALLED_GPU_CCS)
    string(REPLACE " " ";" INSTALLED_GPU_CCS "${INSTALLED_GPU_CCS}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS}")
    message(STATUS "CUDA_ARCH_LIST: ${CUDA_ARCH_LIST}")
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(set_libheom_flags)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LIBHEOM_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${LIBHEOM_FLAGS}")

add_subdirectory(tools)
include(process_jinja2)

include_directories(include)
add_subdirectory(include)
add_subdirectory(src)

# add_subdirectory(3rdparty/msgpack-c)
# set(MSGPACK_USE_BOOST OFF)
# set(msgpack_DIR ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/msgpack-c)
# message(STATUS "${CMAKE_MODULE_PATH}")
# find_package(msgpack REQUIRED)
# include_directories(3rdparty/msgpack-c/include)

add_subdirectory(test)
