/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#include "heom_gpu.h"
#include "utility_gpu.h"
#include "type_gpu.h"
#include "handle_gpu.h"
#include "dense_matrix_gpu.h"
// #include "csr_matrix_gpu.h"
#include "const.h"
#include "cublas_cusparse_wrapper.h"

namespace libheom {


template<typename T, template <typename, int> class MatrixType, int NumState>
class HeomLHGpuVars {
 private:
  HandleGpu handle;
  
  GPU_MATRIX_TYPE(MatrixType)<T> R_heom_impl;
  GPU_MATRIX_TYPE(MatrixType)<T> X_hrchy_impl;
  
  thrust::device_vector<GPU_TYPE(T)> rho;
  thrust::device_vector<GPU_TYPE(T)> sub_vector;

  void CalcDiffGpu(
      thrust::device_vector<GPU_TYPE(T)>& drho_dt,
      const thrust::device_vector<GPU_TYPE(T)>& rho,
      T alpha,
      T beta);

  friend class HeomLHGpu<T, MatrixType, NumState>;
};

template<typename T, template <typename, int> class MatrixType, int NumState>
void HeomLHGpu<T, MatrixType, NumState>::InitAuxVars(std::function<void(int)> callback) {
  HeomLH<T, MatrixType, NumState>::InitAuxVars(callback);

  this->gpu.reset(new HeomLHGpuVars<T, MatrixType, NumState>);
  
  this->gpu->handle.Initialize(device_number);
  
  this->gpu->R_heom_impl = this->R_heom_impl;
  this->gpu->rho.resize(this->size_rho);
  this->gpu->sub_vector.resize(this->size_rho);
}

// template<typename T>
// inline void axpy
// /**/(handle_gpu& handle,
//      T alpha,
//      const thrust::device_vector<GPU_TYPE(T)>* x,
//      thrust::device_vector<GPU_TYPE(T)>& y);

inline void axpy(HandleGpu& handle,
                 complex64 alpha,
                 const thrust::device_vector<GPU_TYPE(complex64)>& x,
                 thrust::device_vector<GPU_TYPE(complex64)>& y) {
  GPU_TYPE(complex64) alpha_gpu = alpha;
  CUBLAS_CALL(cublasCaxpy(handle.cublas,
                          x.size(),
                          raw_gpu_type_cast<const complex64 *>(&alpha_gpu),
                          raw_gpu_type_cast<const complex64 *>(x.data()), 1,
                          raw_gpu_type_cast<complex64 *>(y.data()), 1));
}

inline void axpy(HandleGpu& handle,
                 complex128 alpha,
                 const thrust::device_vector<GPU_TYPE(complex128)>& x,
                 thrust::device_vector<GPU_TYPE(complex128)>& y) {
  GPU_TYPE(complex128) alpha_gpu = alpha;
  CUBLAS_CALL(cublasZaxpy(handle.cublas,
                          x.size(),
                          raw_gpu_type_cast<const complex128 *>(&alpha_gpu),
                          raw_gpu_type_cast<const complex128 *>(x.data()), 1,
                          raw_gpu_type_cast<complex128 *>(y.data()), 1));
}


inline void copy(HandleGpu& handle,
                 const thrust::device_vector<GPU_TYPE(complex64)>& x,
                 thrust::device_vector<GPU_TYPE(complex64)>& y) {
  CUBLAS_CALL(cublasCcopy(handle.cublas,
                          x.size(),
                          raw_gpu_type_cast<const complex64 *>(x.data()), 1,
                          raw_gpu_type_cast<complex64 *>(y.data()), 1));
}


inline void copy(HandleGpu& handle,
                 const thrust::device_vector<GPU_TYPE(complex128)>& x,
                 thrust::device_vector<GPU_TYPE(complex128)>& y) {
  CUBLAS_CALL(cublasZcopy(handle.cublas,
                          x.size(),
                          raw_gpu_type_cast<const complex128 *>(x.data()), 1,
                          raw_gpu_type_cast<complex128 *>(y.data()), 1));
}

template<typename T, template <typename, int> class MatrixType, int NumState>
inline void HeomLHGpuVars<T, MatrixType, NumState>::CalcDiffGpu (
    thrust::device_vector<GPU_TYPE(T)>& drho_dt,
    const thrust::device_vector<GPU_TYPE(T)>& rho,
    T alpha,
    T beta) {
  gemvGpu(this->handle, -alpha, this->R_heom_impl, rho, beta, drho_dt);
}

template<typename T, template <typename, int> class MatrixType, int NumState>
void HeomLHGpu<T, MatrixType, NumState>::CalcDiff(
    Ref<DenseVector<T,Eigen::Dynamic>> drho_dt,
    const Ref<const DenseVector<T,Eigen::Dynamic>>& rho,
    REAL_TYPE(T) alpha,
    REAL_TYPE(T) beta) {
  CopyVectorGpu(rho.data(), this->gpu->rho);
  // thrust::copy_n(reinterpret_cast<const GPU_TYPE(T)*>(rho),
  //                this->gpu->rho.size(),
  //                this->gpu->rho.begin());
  gpu->CalcDiffGpu(this->gpu->sub_vector,
                   this->gpu->rho,
                   alpha,
                   beta);
  CopyVectorGpu(this->gpu->sub_vector, drho_dt.data());
}

template<typename T, template <typename, int> class MatrixType, int NumState>
void HeomLHGpu<T, MatrixType, NumState>::Evolve1(
    Ref<DenseVector<T,Eigen::Dynamic>> rho,
    REAL_TYPE(T) dt) {
  // thrust::host_vector<T> tmp(this->size_rho);
  gpu->CalcDiffGpu(this->gpu->sub_vector,
                   this->gpu->rho,
                   dt,
                   0);
  // tmp = this->gpu->sub_vector;
  // std::ofstream out("drho.dat");
  // for (int i = 0; i < this->size_rho; ++i) {
  //   out << tmp[i] << std::endl;
  // }
  // std::exit(1);
  
  axpy(gpu->handle,
       Frac<T>(1,3),
       this->gpu->sub_vector,
       this->gpu->rho);

  gpu->CalcDiffGpu(this->gpu->sub_vector,
                   this->gpu->rho,
                   dt,
                   -1);
  axpy(gpu->handle,
       Frac<T>(3,4),
       this->gpu->sub_vector,
       this->gpu->rho);
  
  gpu->CalcDiffGpu(this->gpu->sub_vector,
                   this->gpu->rho,
                   dt,
                   -1);
  axpy(gpu->handle,
       Frac<T>(2,3),
       this->gpu->sub_vector,
       this->gpu->rho);
  
  gpu->CalcDiffGpu(this->gpu->sub_vector,
                   this->gpu->rho,
                   dt,
                   -1);
  axpy(gpu->handle,
       Frac<T>(1,4),
       this->gpu->sub_vector,
       this->gpu->rho);
}

template<typename T, template <typename, int> class MatrixType, int NumState>
void HeomLHGpu<T, MatrixType, NumState>::Evolve(
    Ref<DenseVector<T,Eigen::Dynamic>> rho,
    REAL_TYPE(T) dt,
    const int steps) {
  CopyVectorGpu(rho.data(), this->gpu->rho);
  for (int step = 0; step < steps; ++step) {
    Evolve1(rho, dt);
  }
  CopyVectorGpu(this->gpu->rho, rho.data());
};


}

// Explicit instantiations
namespace libheom {

#define DECLARE_EXPLICIT_INSTANTIATIONS(T, MatrixType, NumState)        \
  template class HeomLHGpuVars<T, MatrixType, NumState>;                \
  template void HeomLHGpu<T, MatrixType, NumState>::InitAuxVars(        \
      std::function<void(int)> callback);                               \
  template void HeomLHGpu<T, MatrixType, NumState>::CalcDiff(           \
      Ref<DenseVector<T,Eigen::Dynamic>> drho_dt,                       \
      const Ref<const DenseVector<T,Eigen::Dynamic>>& rho,              \
      REAL_TYPE(T) alpha,                                               \
      REAL_TYPE(T) beta);                                               \
  template void HeomLHGpu<T, MatrixType, NumState>::Evolve(             \
      Ref<DenseVector<T,Eigen::Dynamic>> rho,                           \
      REAL_TYPE(T) dt,                                                  \
      const int steps);                                                 \
  template void HeomLHGpu<T, MatrixType, NumState>::Evolve1(            \
      Ref<DenseVector<T,Eigen::Dynamic>> rho,                           \
      REAL_TYPE(T) dt);

// DECLARE_EXPLICIT_INSTANTIATIONS(complex64,  DenseMatrix);
// DECLARE_EXPLICIT_INSTANTIATIONS(complex64,  CsrMatrix);
DECLARE_EXPLICIT_INSTANTIATIONS(complex128, DenseMatrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(complex128, CsrMatrix,   Eigen::Dynamic);
}
