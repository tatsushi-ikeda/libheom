#  -*- mode:cmake -*-
#  LibHEOM -- C++ unit tests (GoogleTest + CTest)
#  Copyright (c) Tatsushi Ikeda
#  This library is distributed under BSD 3-Clause License.
#  See LICENSE.txt for licence.
# ------------------------------------------------------------------------*/

cmake_minimum_required(VERSION 3.12)

# ---------------------------------------------------------------------------
# Locate GoogleTest
# Hint: GTest may be installed in the active virtual environment
# ---------------------------------------------------------------------------
set(_gtest_hint_paths
    "$ENV{VIRTUAL_ENV}"
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

# GTest may be built against a newer libstdc++ than the system compiler provides.
# If so, add the virtual environment's lib dir to the linker search path and
# embed it as an RPATH so the test binary finds the right libstdc++ at runtime.
set(_venv_libdir "$ENV{VIRTUAL_ENV}/lib")
if(EXISTS "${_venv_libdir}/libstdc++.so.6")
    link_directories("${_venv_libdir}")
    set(_gtest_rpath_flags "-Wl,-rpath,${_venv_libdir}")
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
    # Embed RPATH so the test binary finds the virtual environment's libstdc++ at runtime
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
    add_libheom_gtest(test_hierarchy_space         test_hierarchy_space.cc)
    add_libheom_gtest(test_linalg_engine_eigen test_linalg_engine_eigen.cc)
    add_libheom_gtest(test_heom_construction   test_heom_construction.cc)
    add_libheom_gtest(test_parallel_eigen      test_parallel_eigen.cc)
    add_libheom_gtest(test_heom_dynamics_eigen test_heom_dynamics_eigen.cc)
endif()

# ---------------------------------------------------------------------------
# L_TEST7: MKL backend test
# ---------------------------------------------------------------------------
if(LIBHEOM_ENABLE_MKL_INTERNAL)
    add_libheom_gtest(test_linalg_engine_mkl  test_linalg_engine_mkl.cc)
    add_libheom_gtest(test_heom_dynamics_mkl  test_heom_dynamics_mkl.cc)
endif()

# ---------------------------------------------------------------------------
# L_TEST8: CUDA backend test
# ---------------------------------------------------------------------------
if(LIBHEOM_ENABLE_CUDA_INTERNAL)
    foreach(_cu_test IN ITEMS test_linalg_engine_cuda test_heom_dynamics_cuda)
        add_executable(${_cu_test} ${_cu_test}.cu)
        target_link_libraries(${_cu_test} PRIVATE
            libheom
            GTest::gtest
            GTest::gtest_main
        )
        add_dependencies(${_cu_test} generated_headers)
        set_target_properties(${_cu_test}
            PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
        if(_gtest_rpath_flags)
            target_link_options(${_cu_test} PRIVATE ${_gtest_rpath_flags})
        endif()
        add_test(NAME ${_cu_test} COMMAND ${_cu_test})
    endforeach()
endif()
