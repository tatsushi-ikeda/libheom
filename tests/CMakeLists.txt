#  -*- mode:cmake -*-
#  LibHEOM -- C++ unit tests (GoogleTest + CTest)
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LICENSE.txt for licence.
# ------------------------------------------------------------------------*/

cmake_minimum_required(VERSION 3.12)

# ---------------------------------------------------------------------------
# Locate GoogleTest
# Hint: Intel Python distribution ships GTest 1.14 at ${VIRTUAL_ENV}
# ---------------------------------------------------------------------------
set(_gtest_hint_paths
    "$ENV{VIRTUAL_ENV}"
    "${VIRTUAL_ENV}"
)
list(PREPEND CMAKE_PREFIX_PATH ${_gtest_hint_paths})
find_package(GTest QUIET)
foreach(_p IN LISTS _gtest_hint_paths)
    list(REMOVE_ITEM CMAKE_PREFIX_PATH "${_p}")
endforeach()

if(NOT GTest_FOUND)
    message(STATUS "GoogleTest not found -- C++ unit tests are DISABLED.")
    message(STATUS "  Set CMAKE_PREFIX_PATH to the GTest installation dir to enable.")
    return()
endif()
message(STATUS "GoogleTest ${GTEST_VERSION} found")

# GTest is built against the Intel Python libstdc++ (conda-forge GCC 12.3.0).
# The system compiler (GCC 8.5.0) ships an older libstdc++ that lacks the
# required versioned symbols (GLIBCXX_3.4.26+).  Add Intel Python's lib dir to
# the linker search path and embed it as an RPATH so the newer libstdc++ is used.
set(_intelpython_libdir "${VIRTUAL_ENV}/lib")
if(EXISTS "${_intelpython_libdir}/libstdc++.so.6")
    link_directories("${_intelpython_libdir}")
    set(_gtest_rpath_flags "-Wl,-rpath,${_intelpython_libdir}")
else()
    set(_gtest_rpath_flags "")
endif()

# ---------------------------------------------------------------------------
# Helper: create a GoogleTest executable and register it with CTest
# ---------------------------------------------------------------------------
function(add_libheom_gtest test_name source_file)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} PRIVATE
        libheom
        GTest::gtest
        GTest::gtest_main
    )
    # -UENABLE_CUDA: CMAKE_CXX_FLAGS includes -DENABLE_CUDA globally when CUDA
    # is enabled, but these .cc files are compiled by g++ and cannot include
    # CUDA runtime headers.  Undefine it at the target level.
    target_compile_options(${test_name} PRIVATE -UENABLE_CUDA)
    # Embed RPATH so the test binary finds Intel Python's libstdc++ at runtime
    if(_gtest_rpath_flags)
        target_link_options(${test_name} PRIVATE ${_gtest_rpath_flags})
    endif()
    # Generated Jinja2 headers (linalg_engine_eigen.h etc.) must exist first
    add_dependencies(${test_name} generated_headers)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# ---------------------------------------------------------------------------
# L_TEST2-L_TEST6, L_TEST9: Eigen-backend tests (always built when Eigen enabled)
# ---------------------------------------------------------------------------
if(LIBHEOM_ENABLE_EIGEN_INTERNAL)
    add_libheom_gtest(test_lil_matrix          test_lil_matrix.cc)
    add_libheom_gtest(test_dense_matrix        test_dense_matrix.cc)
    add_libheom_gtest(test_sparse_matrix       test_sparse_matrix.cc)
    add_libheom_gtest(test_hrchy_space         test_hrchy_space.cc)
    add_libheom_gtest(test_linalg_engine_eigen test_linalg_engine_eigen.cc)
    add_libheom_gtest(test_heom_construction   test_heom_construction.cc)
endif()

# ---------------------------------------------------------------------------
# L_TEST7: MKL backend test
# ---------------------------------------------------------------------------
if(LIBHEOM_ENABLE_MKL_INTERNAL)
    add_libheom_gtest(test_linalg_engine_mkl test_linalg_engine_mkl.cc)
endif()

# ---------------------------------------------------------------------------
# L_TEST8: CUDA backend test
# ---------------------------------------------------------------------------
if(LIBHEOM_ENABLE_CUDA_INTERNAL)
    add_executable(test_linalg_engine_cuda test_linalg_engine_cuda.cu)
    target_link_libraries(test_linalg_engine_cuda PRIVATE
        libheom
        GTest::gtest
        GTest::gtest_main
    )
    add_dependencies(test_linalg_engine_cuda generated_headers)
    set_target_properties(test_linalg_engine_cuda
        PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
    if(_gtest_rpath_flags)
        target_link_options(test_linalg_engine_cuda PRIVATE ${_gtest_rpath_flags})
    endif()
    add_test(NAME test_linalg_engine_cuda COMMAND test_linalg_engine_cuda)
endif()
