/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#include "redfield_gpu.h"

#include "utility_gpu.h"
#include "type_gpu.h"
#include "dense_matrix_gpu.h"
#include "handle_gpu.h"
#include "const.h"
#include "cublas_cusparse_wrapper.h"

namespace libheom {

template<typename T, template <typename, int> class MatrixType, int NumState>
class RedfieldHGpuVars {
 private:
  HandleGpu handle;
  
  GPU_MATRIX_TYPE(MatrixType)<T> H_impl;
  std::unique_ptr<GPU_MATRIX_TYPE(MatrixType)<T>[]> V_impl;
  std::unique_ptr<GPU_MATRIX_TYPE(MatrixType)<T>[]> Lambda_impl;
  std::unique_ptr<GPU_MATRIX_TYPE(MatrixType)<T>[]> Lambda_dagger_impl;
  
  thrust::device_vector<GPU_TYPE(T)> rho;
  thrust::device_vector<GPU_TYPE(T)> sub_vector;
  thrust::device_vector<GPU_TYPE(T)> tmp_vector;

  void CalcDiffGpu(RedfieldHGpu<T, MatrixType, NumState>& obj,
                   thrust::device_vector<GPU_TYPE(T)>& drho_dt,
                   const thrust::device_vector<GPU_TYPE(T)>& rho,
                   T alpha,
                   T beta);

  friend class RedfieldHGpu<T, MatrixType, NumState>;
};

template<typename T, template <typename, int> class MatrixType, int NumState>
void RedfieldHGpu<T, MatrixType, NumState>::InitAuxVars(std::function<void(int)> callback) {
  RedfieldH<T, MatrixType, NumState>::InitAuxVars(callback);

  this->gpu.reset(new RedfieldHGpuVars<T, MatrixType, NumState>);
  
  this->gpu->handle.Initialize(device_number);

  this->gpu->H_impl = this->H_impl;
  
  this->gpu->V_impl.reset(new GPU_MATRIX_TYPE(MatrixType)<T>[this->n_noise]);
  this->gpu->Lambda_impl.reset(new GPU_MATRIX_TYPE(MatrixType)<T>[this->n_noise]);
  this->gpu->Lambda_dagger_impl.reset(new GPU_MATRIX_TYPE(MatrixType)<T>[this->n_noise]);
  for (int s = 0; s < this->n_noise; ++s) {
    this->gpu->V_impl[s]      = this->V_impl[s];
    this->gpu->Lambda_impl[s] = this->Lambda_impl[s];
    this->gpu->Lambda_dagger_impl[s] = this->Lambda_dagger_impl[s];
  }
  
  this->gpu->rho.resize(this->size_rho);
  this->gpu->sub_vector.resize(this->size_rho);
  this->gpu->tmp_vector.resize(this->size_rho);
}

inline void axpy (HandleGpu& handle,
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
inline void RedfieldHGpuVars<T, MatrixType, NumState>::CalcDiffGpu(
    RedfieldHGpu<T, MatrixType, NumState>& obj,
    thrust::device_vector<GPU_TYPE(T)>& drho_dt_raw,
    const thrust::device_vector<GPU_TYPE(T)>& rho_raw,
    const T alpha,
    const T beta) {
  DenseMatrixGpuWrapper<T>      drho_dt(obj.n_state, obj.n_state, &drho_dt_raw[0]);
  ConstDenseMatrixGpuWrapper<T> rho(obj.n_state, obj.n_state, &rho_raw[0]);
  DenseMatrixGpuWrapper<T>      tmp(obj.n_state, obj.n_state, &this->tmp_vector[0]);
  gemmGpu(this->handle, -alpha*IUnit<T>(), this->H_impl, rho, static_cast<T>(beta), drho_dt);
  gemmGpu(this->handle, +alpha*IUnit<T>(), rho, this->H_impl, static_cast<T>(1), drho_dt);
  for (int s = 0; s < obj.n_noise; ++s) {
    gemmGpu(this->handle, +IUnit<T>(), this->Lambda_impl[s], rho,  static_cast<T>(0), tmp);
    gemmGpu(this->handle, -IUnit<T>(), rho, this->Lambda_dagger_impl[s],  static_cast<T>(1), tmp);
    gemmGpu(this->handle, +alpha*IUnit<T>(), this->V_impl[s], tmp, static_cast<T>(1), drho_dt);
    gemmGpu(this->handle, -alpha*IUnit<T>(), tmp, this->V_impl[s], static_cast<T>(1), drho_dt);
  }
}

template<typename T, template <typename, int> class MatrixType, int NumState>
void RedfieldHGpu<T, MatrixType, NumState>::CalcDiff(
    Ref<DenseVector<T,Eigen::Dynamic>> drho_dt,
    const Ref<const DenseVector<T,Eigen::Dynamic>>& rho,
    REAL_TYPE(T) alpha,
    REAL_TYPE(T) beta) {
  CopyVectorGpu(rho.data(), this->gpu->rho);
  gpu->CalcDiffGpu(*this,
                   this->gpu->sub_vector,
                   this->gpu->rho,
                   alpha,
                   beta);
  CopyVectorGpu(this->gpu->sub_vector, drho_dt.data());
  
}

template<typename T, template <typename, int> class MatrixType, int NumState>
void RedfieldHGpu<T, MatrixType, NumState>::Evolve1(
    Ref<DenseVector<T,Eigen::Dynamic>> rho,
    REAL_TYPE(T) dt) {
  gpu->CalcDiffGpu(*this,
                   this->gpu->sub_vector,
                   this->gpu->rho,
                   dt,
                   0);
  axpy(gpu->handle,
       Frac<T>(1,3),
       this->gpu->sub_vector,
       this->gpu->rho);

  gpu->CalcDiffGpu(*this,
                        this->gpu->sub_vector,
                        this->gpu->rho,
                        dt,
                        -1);
  axpy(gpu->handle,
       Frac<T>(3,4),
       this->gpu->sub_vector,
       this->gpu->rho);
  
  gpu->CalcDiffGpu(*this,
                        this->gpu->sub_vector,
                        this->gpu->rho,
                        dt,
                        -1);
  axpy(gpu->handle,
       Frac<T>(2,3),
       this->gpu->sub_vector,
       this->gpu->rho);
  
  gpu->CalcDiffGpu(*this,
                        this->gpu->sub_vector,
                        this->gpu->rho,
                        dt,
                        -1);
  axpy(gpu->handle,
       Frac<T>(1,4),
       this->gpu->sub_vector,
       this->gpu->rho);
}

template<typename T, template <typename, int> class MatrixType, int NumState>
void RedfieldHGpu<T, MatrixType, NumState>::Evolve(
    Ref<DenseVector<T,Eigen::Dynamic>> rho,
    REAL_TYPE(T) dt,
    const int steps){
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
  template class RedfieldHGpuVars<T, MatrixType, NumState>;             \
  template void RedfieldHGpu<T, MatrixType, NumState>::InitAuxVars(     \
      std::function<void(int)> callback);                               \
  template void RedfieldHGpu<T, MatrixType, NumState>::CalcDiff(        \
      Ref<DenseVector<T,Eigen::Dynamic>> drho_dt,                       \
      const Ref<const DenseVector<T,Eigen::Dynamic>>& rho,              \
      REAL_TYPE(T) alpha,                                               \
      REAL_TYPE(T) beta);                                               \
  template void RedfieldHGpu<T, MatrixType, NumState>::Evolve(          \
      Ref<DenseVector<T,Eigen::Dynamic>> rho,                           \
      REAL_TYPE(T) dt,                                                  \
      const int steps);                                                 \
  template void RedfieldHGpu<T, MatrixType, NumState>::Evolve1(         \
      Ref<DenseVector<T,Eigen::Dynamic>> rho,                           \
      REAL_TYPE(T) dt);

DECLARE_EXPLICIT_INSTANTIATIONS(complex64,  DenseMatrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(complex64,  CsrMatrix,   Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(complex128, DenseMatrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(complex128, CsrMatrix,   Eigen::Dynamic);
}

