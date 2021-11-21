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

template<typename T, template <typename, int> class matrix_type, int num_state>
class redfield_h_gpu_vars {
 private:
  handle_gpu handle;
  
  GPU_MATRIX_TYPE(matrix_type)<T> H_impl;
  std::unique_ptr<GPU_MATRIX_TYPE(matrix_type)<T>[]> V_impl;
  std::unique_ptr<GPU_MATRIX_TYPE(matrix_type)<T>[]> Lambda_impl;
  std::unique_ptr<GPU_MATRIX_TYPE(matrix_type)<T>[]> Lambda_dagger_impl;
  
  thrust::device_vector<GPU_TYPE(T)> rho;
  thrust::device_vector<GPU_TYPE(T)> sub_vector;
  thrust::device_vector<GPU_TYPE(T)> tmp_vector;

  void calc_diff_gpu(redfield_h_gpu<T, matrix_type, num_state>& obj,
                     thrust::device_vector<GPU_TYPE(T)>& drho_dt,
                     const thrust::device_vector<GPU_TYPE(T)>& rho,
                     T alpha,
                     T beta);

  friend class redfield_h_gpu<T, matrix_type, num_state>;
};

template<typename T, template <typename, int> class matrix_type, int num_state>
void redfield_h_gpu<T, matrix_type, num_state>::init_aux_vars(std::function<void(int)> callback) {
  redfield_h<T, matrix_type, num_state>::init_aux_vars(callback);

  this->gpu.reset(new redfield_h_gpu_vars<T, matrix_type, num_state>);
  
  this->gpu->handle.Initialize(device_number);

  this->gpu->H_impl = this->H_impl;
  
  this->gpu->V_impl.reset(new GPU_MATRIX_TYPE(matrix_type)<T>[this->n_noise]);
  this->gpu->Lambda_impl.reset(new GPU_MATRIX_TYPE(matrix_type)<T>[this->n_noise]);
  this->gpu->Lambda_dagger_impl.reset(new GPU_MATRIX_TYPE(matrix_type)<T>[this->n_noise]);
  for (int s = 0; s < this->n_noise; ++s) {
    this->gpu->V_impl[s]      = this->V_impl[s];
    this->gpu->Lambda_impl[s] = this->Lambda_impl[s];
    this->gpu->Lambda_dagger_impl[s] = this->Lambda_dagger_impl[s];
  }
  
  this->gpu->rho.resize(this->size_rho);
  this->gpu->sub_vector.resize(this->size_rho);
  this->gpu->tmp_vector.resize(this->size_rho);
}

inline void axpy (handle_gpu& handle,
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


inline void axpy(handle_gpu& handle,
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


inline void copy(handle_gpu& handle,
                 const thrust::device_vector<GPU_TYPE(complex64)>& x,
                 thrust::device_vector<GPU_TYPE(complex64)>& y) {
  CUBLAS_CALL(cublasCcopy(handle.cublas,
                          x.size(),
                          raw_gpu_type_cast<const complex64 *>(x.data()), 1,
                          raw_gpu_type_cast<complex64 *>(y.data()), 1));
}


inline void copy(handle_gpu& handle,
                 const thrust::device_vector<GPU_TYPE(complex128)>& x,
                 thrust::device_vector<GPU_TYPE(complex128)>& y) {
  CUBLAS_CALL(cublasZcopy(handle.cublas,
                          x.size(),
                          raw_gpu_type_cast<const complex128 *>(x.data()), 1,
                          raw_gpu_type_cast<complex128 *>(y.data()), 1));
}


template<typename T, template <typename, int> class matrix_type, int num_state>
inline void redfield_h_gpu_vars<T, matrix_type, num_state>::calc_diff_gpu(
    redfield_h_gpu<T, matrix_type, num_state>& obj,
    thrust::device_vector<GPU_TYPE(T)>& drho_dt_raw,
    const thrust::device_vector<GPU_TYPE(T)>& rho_raw,
    const T alpha,
    const T beta) {
  dense_matrix_gpu_wrapper<T>       drho_dt(obj.n_state, obj.n_state, &drho_dt_raw[0]);
  const_dense_matrix_gpu_wrapper<T> rho(obj.n_state, obj.n_state, &rho_raw[0]);
  dense_matrix_gpu_wrapper<T>       tmp(obj.n_state, obj.n_state, &this->tmp_vector[0]);
  gemm_gpu(this->handle, -alpha*i_unit<T>(), this->H_impl, rho, static_cast<T>(beta), drho_dt);
  gemm_gpu(this->handle, +alpha*i_unit<T>(), rho, this->H_impl, static_cast<T>(1), drho_dt);
  for (int s = 0; s < obj.n_noise; ++s) {
    gemm_gpu(this->handle, +i_unit<T>(), this->Lambda_impl[s], rho,  static_cast<T>(0), tmp);
    gemm_gpu(this->handle, -i_unit<T>(), rho, this->Lambda_dagger_impl[s],  static_cast<T>(1), tmp);
    gemm_gpu(this->handle, +alpha*i_unit<T>(), this->V_impl[s], tmp, static_cast<T>(1), drho_dt);
    gemm_gpu(this->handle, -alpha*i_unit<T>(), tmp, this->V_impl[s], static_cast<T>(1), drho_dt);
  }
}

template<typename T, template <typename, int> class matrix_type, int num_state>
void redfield_h_gpu<T, matrix_type, num_state>::calc_diff(
    ref<DenseVector<T,Eigen::Dynamic>> drho_dt,
    const ref<const DenseVector<T,Eigen::Dynamic>>& rho,
    REAL_TYPE(T) alpha,
    REAL_TYPE(T) beta) {
  copy_vector_gpu(rho.data(), this->gpu->rho);
  gpu->calc_diff_gpu(*this,
                   this->gpu->sub_vector,
                   this->gpu->rho,
                   alpha,
                   beta);
  copy_vector_gpu(this->gpu->sub_vector, drho_dt.data());
  
}

template<typename T, template <typename, int> class matrix_type, int num_state>
void redfield_h_gpu<T, matrix_type, num_state>::evolve_1(
    ref<DenseVector<T,Eigen::Dynamic>> rho,
    REAL_TYPE(T) dt) {
  gpu->calc_diff_gpu(*this,
                   this->gpu->sub_vector,
                   this->gpu->rho,
                   dt,
                   0);
  axpy(gpu->handle,
       frac<T>(1,3),
       this->gpu->sub_vector,
       this->gpu->rho);

  gpu->calc_diff_gpu(*this,
                        this->gpu->sub_vector,
                        this->gpu->rho,
                        dt,
                        -1);
  axpy(gpu->handle,
       frac<T>(3,4),
       this->gpu->sub_vector,
       this->gpu->rho);
  
  gpu->calc_diff_gpu(*this,
                        this->gpu->sub_vector,
                        this->gpu->rho,
                        dt,
                        -1);
  axpy(gpu->handle,
       frac<T>(2,3),
       this->gpu->sub_vector,
       this->gpu->rho);
  
  gpu->calc_diff_gpu(*this,
                        this->gpu->sub_vector,
                        this->gpu->rho,
                        dt,
                        -1);
  axpy(gpu->handle,
       frac<T>(1,4),
       this->gpu->sub_vector,
       this->gpu->rho);
}

template<typename T, template <typename, int> class matrix_type, int num_state>
void redfield_h_gpu<T, matrix_type, num_state>::evolve(
    ref<DenseVector<T,Eigen::Dynamic>> rho,
    REAL_TYPE(T) dt,
    const int steps){
  copy_vector_gpu(rho.data(), this->gpu->rho);
  for (int step = 0; step < steps; ++step) {
    evolve_1(rho, dt);
  }
  copy_vector_gpu(this->gpu->rho, rho.data());
};

}

// Explicit instantiations
namespace libheom {

#define DECLARE_EXPLICIT_INSTANTIATIONS(T, matrix_type, num_state)        \
  template class redfield_h_gpu_vars<T, matrix_type, num_state>;             \
  template void redfield_h_gpu<T, matrix_type, num_state>::init_aux_vars(     \
      std::function<void(int)> callback);                               \
  template void redfield_h_gpu<T, matrix_type, num_state>::calc_diff(        \
      ref<DenseVector<T,Eigen::Dynamic>> drho_dt,                       \
      const ref<const DenseVector<T,Eigen::Dynamic>>& rho,              \
      REAL_TYPE(T) alpha,                                               \
      REAL_TYPE(T) beta);                                               \
  template void redfield_h_gpu<T, matrix_type, num_state>::evolve(          \
      ref<DenseVector<T,Eigen::Dynamic>> rho,                           \
      REAL_TYPE(T) dt,                                                  \
      const int steps);                                                 \
  template void redfield_h_gpu<T, matrix_type, num_state>::evolve_1(         \
      ref<DenseVector<T,Eigen::Dynamic>> rho,                           \
      REAL_TYPE(T) dt);

DECLARE_EXPLICIT_INSTANTIATIONS(complex64,  dense_matrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(complex64,  csr_matrix,   Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(complex128, dense_matrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(complex128, csr_matrix,   Eigen::Dynamic);
}

