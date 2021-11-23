/*
 * LibHEOM: Copyright (c) Tatsushi Ikeda
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


template<typename T, template <typename, int> class matrix_type, int num_state>
class heom_lh_gpu_vars {
 private:
  handle_gpu handle;
  
  GPU_MATRIX_TYPE(matrix_type)<T> R_heom_impl;
  GPU_MATRIX_TYPE(matrix_type)<T> X_hrchy_impl;
  
  thrust::device_vector<GPU_TYPE(T)> rho;
  thrust::device_vector<GPU_TYPE(T)> sub_vector;

  void calc_diff_gpu(
      thrust::device_vector<GPU_TYPE(T)>& drho_dt,
      const thrust::device_vector<GPU_TYPE(T)>& rho,
      T alpha,
      T beta);

  friend class heom_lh_gpu<T, matrix_type, num_state>;
};

template<typename T, template <typename, int> class matrix_type, int num_state>
void heom_lh_gpu<T, matrix_type, num_state>::init_aux_vars(std::function<void(int)> callback) {
  heom_lh<T, matrix_type, num_state>::init_aux_vars(callback);

  this->gpu.reset(new heom_lh_gpu_vars<T, matrix_type, num_state>);
  
  this->gpu->handle.init(device_number);
  
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

inline void axpy(handle_gpu& handle,
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
inline void heom_lh_gpu_vars<T, matrix_type, num_state>::calc_diff_gpu (
    thrust::device_vector<GPU_TYPE(T)>& drho_dt,
    const thrust::device_vector<GPU_TYPE(T)>& rho,
    T alpha,
    T beta) {
  gemv_gpu(this->handle, -alpha, this->R_heom_impl, rho, beta, drho_dt);
}

template<typename T, template <typename, int> class matrix_type, int num_state>
void heom_lh_gpu<T, matrix_type, num_state>::calc_diff(
    ref<dense_vector<T,Eigen::Dynamic>> drho_dt,
    const ref<const dense_vector<T,Eigen::Dynamic>>& rho,
    REAL_TYPE(T) alpha,
    REAL_TYPE(T) beta) {
  copy_vector_gpu(rho.data(), this->gpu->rho);
  // thrust::copy_n(reinterpret_cast<const GPU_TYPE(T)*>(rho),
  //                this->gpu->rho.size(),
  //                this->gpu->rho.begin());
  gpu->calc_diff_gpu(this->gpu->sub_vector,
                     this->gpu->rho,
                     alpha,
                     beta);
  copy_vector_gpu(this->gpu->sub_vector, drho_dt.data());
}

template<typename T, template <typename, int> class matrix_type, int num_state>
void heom_lh_gpu<T, matrix_type, num_state>::evolve_1(
    ref<dense_vector<T,Eigen::Dynamic>> rho,
    REAL_TYPE(T) dt) {
  // thrust::host_vector<T> tmp(this->size_rho);
  gpu->calc_diff_gpu(this->gpu->sub_vector,
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
       frac<T>(1,3),
       this->gpu->sub_vector,
       this->gpu->rho);

  gpu->calc_diff_gpu(this->gpu->sub_vector,
                   this->gpu->rho,
                   dt,
                   -1);
  axpy(gpu->handle,
       frac<T>(3,4),
       this->gpu->sub_vector,
       this->gpu->rho);
  
  gpu->calc_diff_gpu(this->gpu->sub_vector,
                     this->gpu->rho,
                     dt,
                     -1);
  axpy(gpu->handle,
       frac<T>(2,3),
       this->gpu->sub_vector,
       this->gpu->rho);
  
  gpu->calc_diff_gpu(this->gpu->sub_vector,
                     this->gpu->rho,
                     dt,
                     -1);
  axpy(gpu->handle,
       frac<T>(1,4),
       this->gpu->sub_vector,
       this->gpu->rho);
}

template<typename T, template <typename, int> class matrix_type, int num_state>
void heom_lh_gpu<T, matrix_type, num_state>::evolve(
    ref<dense_vector<T,Eigen::Dynamic>> rho,
    REAL_TYPE(T) dt,
    const int steps) {
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
  template class heom_lh_gpu_vars<T, matrix_type, num_state>;                \
  template void heom_lh_gpu<T, matrix_type, num_state>::init_aux_vars(        \
      std::function<void(int)> callback);                               \
  template void heom_lh_gpu<T, matrix_type, num_state>::calc_diff(           \
      ref<dense_vector<T,Eigen::Dynamic>> drho_dt,                       \
      const ref<const dense_vector<T,Eigen::Dynamic>>& rho,              \
      REAL_TYPE(T) alpha,                                               \
      REAL_TYPE(T) beta);                                               \
  template void heom_lh_gpu<T, matrix_type, num_state>::evolve(             \
      ref<dense_vector<T,Eigen::Dynamic>> rho,                           \
      REAL_TYPE(T) dt,                                                  \
      const int steps);                                                 \
  template void heom_lh_gpu<T, matrix_type, num_state>::evolve_1(            \
      ref<dense_vector<T,Eigen::Dynamic>> rho,                           \
      REAL_TYPE(T) dt);

// DECLARE_EXPLICIT_INSTANTIATIONS(complex64,  DenseMatrix);
// DECLARE_EXPLICIT_INSTANTIATIONS(complex64,  CsrMatrix);
DECLARE_EXPLICIT_INSTANTIATIONS(complex128, DenseMatrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(complex128, CsrMatrix,   Eigen::Dynamic);
}
