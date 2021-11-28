cmake_minimum_required(VERSION 3.9)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

string(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_LOWER)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}\
    -DBUILD_TYPE=${CMAKE_BUILD_TYPE_LOWER}")

file(READ "VERSION.txt" VER)
string(STRIP "${VER}" VER)
project(libheom VERSION "${VER}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}\
    -DVERSION_MAJOR=${PROJECT_VERSION_MAJOR}\
    -DVERSION_MINOR=${PROJECT_VERSION_MINOR}\
    -DVERSION_PATCH=${PROJECT_VERSION_PATCH}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# prepare gpgpu option
set(ENABLE_GPU_PARALLELIZATION AUTO CACHE STRING "Enable GPU parallelization support")
set_property(CACHE ENABLE_GPU_PARALLELIZATION PROPERTY STRINGS AUTO ON OFF)
if(ENABLE_GPU_PARALLELIZATION STREQUAL "AUTO")
  find_program(NVCC_PATH nvcc)
  if(NVCC_PATH STREQUAL "NVCC_PATH-NOTFOUND")
    set(ENABLE_GPU_PARALLELIZATION OFF)
  else()
    set(ENABLE_GPU_PARALLELIZATION ON)
  endif()
endif()

if(ENABLE_GPU_PARALLELIZATION)
  find_package(CUDA REQUIRED)
  if(WIN32)
  else()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC")
    endif()
  endif()
endif()

# eigen library
if(EXISTS "${CMAKE_SOURCE_DIR}/3rdparty/Eigen")
  include_directories(3rdparty/Eigen)
  message(STATUS "Found Eigen Library in 3rdparty/ directory")
else()
  message(FATAL_ERROR "Could not find Eigen Library")
endif()

# pybind11 library
if(EXISTS "${CMAKE_SOURCE_DIR}/3rdparty/pybind11")
  add_subdirectory(3rdparty/pybind11)
  message(STATUS "Found pybind11 Library in 3rdparty/ directory")
else()
  find_package(pybind11 REQUIRED)
endif()

string(TOLOWER ${CMAKE_CXX_COMPILER_ID} CMAKE_CXX_COMPILER_ID_LOWER)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}\
    -DCOMPILER_NAME=${CMAKE_CXX_COMPILER_ID_LOWER}\
    -DCOMPILER_VERSION=${CMAKE_CXX_COMPILER_VERSION}")

if(CMAKE_VERSION VERSION_LESS "3.1")
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -static")
  endif()
  if(ENABLE_GPU_PARALLELIZATION)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
  endif()
else()
  set(CMAKE_CXX_STANDARD 11)
  if(ENABLE_GPU_PARALLELIZATION)
    set(CMAKE_CUDA_STANDARD 11)
  endif()
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4819")
endif()

include(CheckCXXCompilerFlag)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS_RELEASE "-fast -DNDEBUG")
endif()
# set(CMAKE_CXX_FLAGS "/Ox /Ob2 /fp:fast ${CMAKE_CXX_FLAGS_RELEASE}")


# Specify the architecture for code generation
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xHost")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  CHECK_CXX_COMPILER_FLAG("/arch:AVX2"    COMPILER_OPT_ARCH_AVX2_SUPPORTED)
  CHECK_CXX_COMPILER_FLAG("/arch:AVX"     COMPILER_OPT_ARCH_AVX_SUPPORTED)
  if(COMPILER_OPT_ARCH_AVX2_SUPPORTED)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX2")
  elseif(COMPILER_OPT_ARCH_AVX_SUPPORTED)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX")
  endif()
else()
  CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
  if(COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
  endif()
endif()
CHECK_CXX_COMPILER_FLAG("-fabi-version=0" COMPILER_OPT_FABI_VERSION_SUPPORTED)
if(COMPILER_OPT_FABI_VERSION_SUPPORTED)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fabi-version=0")
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(ENABLE_MKL AUTO CACHE STRING "Enable MKL support")
set_property(CACHE ENABLE_MKL PROPERTY STRINGS AUTO ON OFF)
message(STATUS "ENABLE_MKL=${ENABLE_MKL}")

if(NOT (ENABLE_MKL STREQUAL "OFF"))
  set(BLA_VENDOR Intel10_64lp)
  find_package(BLAS)
endif()

if(ENABLE_MKL STREQUAL "ON")
  if(NOT BLAS_FOUND)
    message(FATAL_ERROR "Could not find Intel(R) Math Kernel Library while ENABLE_MKL=${ENABLE_MKL}")
  endif()
endif()

if(BLAS_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_SUPPORT=enabled")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_SUPPORT=disabled")
endif()

include_directories(include)

add_subdirectory(src)
file(INSTALL include DESTINATION ${CMAKE_BINARY_DIR})


