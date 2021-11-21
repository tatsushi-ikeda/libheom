/*
 * LibHEOM: Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#include "qme.h"

#include "const.h"

// Note: In order to avoid compile errors caused by the two phase name
//       lookup rule regarding template class, all class variables in
//       this source have this-> modifier.

namespace libheom {

template<typename T>
void qme<T>::allocate_noise(int n_noise) {
  this->n_noise = n_noise;
  
  this->V.reset(new lil_matrix<T>[n_noise]);

  this->len_gamma.reset(new int [n_noise]);
  this->gamma.reset(new Eigen::SparseMatrix<T, Eigen::RowMajor>[n_noise]);
  this->phi_0.reset(new Eigen::Matrix<T,Eigen::Dynamic,1>[n_noise]);
  this->sigma.reset(new Eigen::Matrix<T,Eigen::Dynamic,1>[n_noise]);
  
  this->s.reset(new Eigen::SparseMatrix<T, Eigen::RowMajor>[n_noise]);
  this->S_delta.reset(new T [n_noise]);
  this->a.reset(new Eigen::SparseMatrix<T, Eigen::RowMajor>[n_noise]);
}


template<typename T>
void qme<T>::initialize() {
  this->size_rho = this->n_state*this->n_state;
  this->sub_vector.resize(this->size_rho);
}

template<typename T>
void qme<T>::finalize() {
}

template<typename T>
void qme<T>::time_evolution(ref<dense_vector<T,Eigen::Dynamic>> rho,
                           REAL_TYPE(T) dt__unit,
                           REAL_TYPE(T) dt,
                           int interval,
                           int count,
                           std::function<void(REAL_TYPE(T))> callback) {
  for (int ctr = 0; ctr < count; ++ctr) {
    callback(ctr*interval*dt__unit);
    evolve(rho, dt, interval);
  }
}


template<typename T>
void qme<T>::evolve_1(ref<dense_vector<T,Eigen::Dynamic>> rho, REAL_TYPE(T) dt) {
  calc_diff(this->sub_vector, rho, dt, 0);
  rho.noalias() += frac<REAL_TYPE(T)>(1,3)*this->sub_vector;

  calc_diff(this->sub_vector, rho, dt, -1);
  rho.noalias() += frac<REAL_TYPE(T)>(3,4)*this->sub_vector;
  
  calc_diff(this->sub_vector, rho, dt, -1);
  rho.noalias() += frac<REAL_TYPE(T)>(2,3)*this->sub_vector;
  
  calc_diff(this->sub_vector, rho, dt, -1);
  rho.noalias() += frac<REAL_TYPE(T)>(1,4)*this->sub_vector;
}

template<typename T>
void qme<T>::evolve(ref<dense_vector<T,Eigen::Dynamic>> rho, REAL_TYPE(T) dt, const int steps) {
  for (int step = 0; step < steps; ++step) {
    evolve_1(rho, dt);
  }
}

}

// Explicit instantiations
namespace libheom {

#define DECLARE_EXPLICIT_INSTANTIATIONS(T)                                    \
  template void qme<T>::allocate_noise(int n_noise);                           \
  template void qme<T>::initialize();                                         \
  template void qme<T>::finalize();                                           \
  template void qme<T>::time_evolution(                                        \
      ref<dense_vector<T,Eigen::Dynamic>> rho,                                                    \
      REAL_TYPE(T) dt__unit,                                                  \
      REAL_TYPE(T) dt,                                                        \
      int interval,                                                           \
      int count,                                                              \
      std::function<void(REAL_TYPE(T))> callback);                            \
  template void qme<T>::evolve(ref<dense_vector<T,Eigen::Dynamic>> rho,                           \
                               REAL_TYPE(T) dt,                         \
                               const int steps);                              \
  template void qme<T>::evolve_1(ref<dense_vector<T,Eigen::Dynamic>> rho, REAL_TYPE(T) dt);

DECLARE_EXPLICIT_INSTANTIATIONS(complex64);
DECLARE_EXPLICIT_INSTANTIATIONS(complex128);

}
