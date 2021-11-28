set(LIBHEOM_SRC
  hrchy_space.cc
  qme.cc
  heom.cc
  redfield.cc
  printer.cc
  # mkl_wrapper.cc
  )

if(ENABLE_GPU_PARALLELIZATION)
  list(APPEND LIBHEOM_SRC
    heom_gpu.cu
    redfield_gpu.cu
    handle_gpu.cu
    utility_gpu.cu
    gpu_info.cu)
  set(CUDA_LIBRARIES
    cuda
    cudart
    cublas
    cusparse)
else()
  list(APPEND LIBHEOM_SRC
    gpu_info.cc)
endif()

if(ENABLE_GPU_PARALLELIZATION)
  cuda_add_library(libheom STATIC ${LIBHEOM_SRC})
else()
  add_library(libheom STATIC ${LIBHEOM_SRC})
endif()

if(ENABLE_GPU_PARALLELIZATION)
  target_compile_definitions(libheom PUBLIC SUPPORT_GPU_PARALLELIZATION)
endif()

# get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
# foreach(dir ${dirs})
#   message(STATUS "dir='${dir}'")
# endforeach()

if(BLAS_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_USE_MKL_ALL")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
  target_link_libraries(libheom PUBLIC "${BLAS_LIBRARIES}")
endif()

if(ENABLE_GPU_PARALLELIZATION)
  target_link_libraries(libheom PRIVATE ${CUDA_LIBRARIES})
else()
endif()

# add_library(libheom::libheom ALAS libheom)
# install(TARGETS libheom
#         LIBRARY DESTINATION lib
#         ARCHIVE DESTINATION lib)

pybind11_add_module(pylibheom pylibheom.cc)
target_link_libraries(pylibheom PRIVATE libheom)

set_property(TARGET pylibheom PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
