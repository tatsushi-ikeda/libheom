#  -*- mode:cmake -*-
#  LibHEOM
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LINCENSE.txt for licence.
# ------------------------------------------------------------------------*/

cmake_minimum_required(VERSION 3.12)

file(READ "VERSION.txt" VER)
string(STRIP "${VER}" VER)

project(libheom)
set(PROJECT_VERSION ${VER})

add_subdirectory(cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR}/cmake)

set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

include(libheom-common)
include(libheom-default-compiler-setting)

include_directories(${LIBHEOM_INCLUDES})

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  ${LIBHEOM_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${LIBHEOM_CUDA_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS       "${CMAKE_SHARED_LINKER_FLAGS} ${LIBHEOM_SHARED_LINKER_FLAGS}")
set(CMAKE_SHARED_LIBRARY_CUDA_FLAGS "${CMAKE_SHARED_LIBRARY_CUDA_FLAGS} ${LIBHEOM_SHARED_LINKER_FLAGS}")

add_subdirectory(tools)
add_subdirectory(include)
add_subdirectory(src)

if(EXISTS "${PROJECT_SOURCE_DIR}/test")
  add_subdirectory(test)
endif()
