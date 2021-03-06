cmake_minimum_required(VERSION 3.9)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

project(libheom_dev
        VERSION 0.5.1)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Prepare GPGPU option
set(ENABLE_GPU_PARALLELIZATION AUTO CACHE STRING "Enable GPU parallelization support")
set_property(CACHE ENABLE_GPU_PARALLELIZATION PROPERTY STRINGS AUTO ON OFF)
if(ENABLE_GPU_PARALLELIZATION STREQUAL "AUTO")
  find_program(NVCC_PATH nvcc)
  if (NVCC_PATH STREQUAL "NVCC_PATH-NOTFOUND")
    set(ENABLE_GPU_PARALLELIZATION OFF)
  else ()
    set(ENABLE_GPU_PARALLELIZATION ON)
  endif ()
endif ()

if(ENABLE_GPU_PARALLELIZATION)
  find_package(CUDA REQUIRED)
  if (WIN32)
  else ()
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC")
    endif ()
  endif ()
endif()

# Eigen Library
if(EXISTS "${CMAKE_SOURCE_DIR}/3rdparty/Eigen")
  include_directories(3rdparty/Eigen)
  message(STATUS "Found Eigen Library in 3rdparty/ directory")
endif()

# pybind11 Library
if(EXISTS "${CMAKE_SOURCE_DIR}/3rdparty/pybind11")
  add_subdirectory(3rdparty/pybind11)
  message(STATUS "Found pybind11 Library in 3rdparty/ directory")
else()
  find_package(pybind11 REQUIRED)  
endif()

if (CMAKE_VERSION VERSION_LESS "3.1")
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
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



if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4819")
endif()

include(CheckCXXCompilerFlag)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS_RELEASE "-fast -DNDEBUG")
endif ()
# set(CMAKE_CXX_FLAGS "/Ox /Ob2 /fp:fast ${CMAKE_CXX_FLAGS_RELEASE}")


# Specify the architecture for code generation
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xHost")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  CHECK_CXX_COMPILER_FLAG("/arch:AVX2"    COMPILER_OPT_ARCH_AVX2_SUPPORTED)
  CHECK_CXX_COMPILER_FLAG("/arch:AVX"     COMPILER_OPT_ARCH_AVX_SUPPORTED)
  if (COMPILER_OPT_ARCH_AVX2_SUPPORTED)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX2")
  elseif (COMPILER_OPT_ARCH_AVX_SUPPORTED)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX")
  endif ()
else ()
  CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
  if (COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
  endif ()
endif ()
CHECK_CXX_COMPILER_FLAG("-fabi-version=0" COMPILER_OPT_FABI_VERSION_SUPPORTED)
if (COMPILER_OPT_FABI_VERSION_SUPPORTED)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fabi-version=0")
endif ()

# Detect OpenMP options.
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # This code is for old Cmake (2.8.12.2), in which OpenMP option for Intel compiler is not treated properly.
  set(OpenMP_CXX_FLAGS "-qopenmp")
else()
  find_package(OpenMP REQUIRED)
  if(OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

# Find the BLAS stuff
# list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# set(BLA_DEFINITIONS)

# if ("$ENV{MKLROOT}" STREQUAL "")
#   message(FATAL_ERROR "MKLROOT is not set. Intel(R) MKL is not detected.")
#   # set(USE_MKL OFF)
# else()
#   set(USE_MKL ON)
#   # message(STATUS "MKLROOT is not set. Please source the Intel MKL mklvars.sh file.")
# endif()

# if (USE_MKL)
#   option(MKL_USE_parallel "Use MKL parallel" True)
#   option(MKL_USE_sdl "Single Dynamic Library or static/dynamic" False)
#   option(BLA_STATIC "Single Static Library" True)
#   set(MKL_USE_interface "lp64" CACHE STRING "for Intel(R)64 compatible arch: ilp64/lp64 or for ia32 arch: cdecl/stdcall")

#   if (BLA_VENDOR MATCHES "_seq")
#     set(MKL_USE_parallel OFF)
#   else()
#     set(MKL_USE_parallel ON)
#   endif()

#   find_package(MKL REQUIRED)
# endif()

# user defined options for MKL

# if (MKL_FOUND)
#     set(BLA_INCLUDED_IR ${MKL_INCLUDE_DIR})
#     set(BLAS_LIBRARIES ${MKL_LIBRARIES})
#     set(BLA_DEFINITIONS USE_MKL)
# endif()

# message(STATUS "BLAS Libraries: ${BLAS_LIBRARIES}")

include_directories(include)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_subdirectory(src)
file(INSTALL include DESTINATION ${CMAKE_BINARY_DIR})

# add_subdirectory(test)

