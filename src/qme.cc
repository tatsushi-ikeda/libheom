/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
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
void Qme<T>::AllocateNoise(int n_noise) {
  this->n_noise = n_noise;
  
  this->V.reset(new LilMatrix<T>[n_noise]);

  this->len_gamma.reset(new int [n_noise]);
  this->gamma.reset(new Eigen::SparseMatrix<T, Eigen::RowMajor>[n_noise]);
  this->phi_0.reset(new Eigen::Matrix<T,Eigen::Dynamic,1>[n_noise]);
  this->sigma.reset(new Eigen::Matrix<T,Eigen::Dynamic,1>[n_noise]);
  
  this->s.reset(new Eigen::SparseMatrix<T, Eigen::RowMajor>[n_noise]);
  this->S_delta.reset(new T [n_noise]);
  this->a.reset(new Eigen::SparseMatrix<T, Eigen::RowMajor>[n_noise]);
}


template<typename T>
void Qme<T>::Initialize() {
  this->size_rho = this->n_state*this->n_state;
  this->sub_vector.resize(this->size_rho);
}

template<typename T>
void Qme<T>::Finalize() {
}

template<typename T>
void Qme<T>::TimeEvolution(Eigen::Ref<DenseVector<T,Eigen::Dynamic>> rho,
                           REAL_TYPE(T) dt__unit,
                           REAL_TYPE(T) dt,
                           int interval,
                           int count,
                           std::function<void(REAL_TYPE(T))> callback) {
  for (int ctr = 0; ctr < count; ++ctr) {
    callback(ctr*interval*dt__unit);
    Evolve(rho, dt, interval);
  }
}


template<typename T>
void Qme<T>::Evolve1(Eigen::Ref<DenseVector<T,Eigen::Dynamic>> rho, REAL_TYPE(T) dt) {
  CalcDiff(this->sub_vector, rho, dt, 0);
  rho.noalias() += Frac<REAL_TYPE(T)>(1,3)*this->sub_vector;

  CalcDiff(this->sub_vector, rho, dt, -1);
  rho.noalias() += Frac<REAL_TYPE(T)>(3,4)*this->sub_vector;
  
  CalcDiff(this->sub_vector, rho, dt, -1);
  rho.noalias() += Frac<REAL_TYPE(T)>(2,3)*this->sub_vector;
  
  CalcDiff(this->sub_vector, rho, dt, -1);
  rho.noalias() += Frac<REAL_TYPE(T)>(1,4)*this->sub_vector;
}

template<typename T>
void Qme<T>::Evolve(Eigen::Ref<DenseVector<T,Eigen::Dynamic>> rho, REAL_TYPE(T) dt, const int steps) {
  for (int step = 0; step < steps; ++step) {
    Evolve1(rho, dt);
  }
}

}

// Explicit instantiations
namespace libheom {

#define DECLARE_EXPLICIT_INSTANTIATIONS(T)                                    \
  template void Qme<T>::AllocateNoise(int n_noise);                           \
  template void Qme<T>::Initialize();                                         \
  template void Qme<T>::Finalize();                                           \
  template void Qme<T>::TimeEvolution(                                        \
      Eigen::Ref<DenseVector<T,Eigen::Dynamic>> rho,                                                    \
      REAL_TYPE(T) dt__unit,                                                  \
      REAL_TYPE(T) dt,                                                        \
      int interval,                                                           \
      int count,                                                              \
      std::function<void(REAL_TYPE(T))> callback);                            \
  template void Qme<T>::Evolve(Eigen::Ref<DenseVector<T,Eigen::Dynamic>> rho,                           \
                               REAL_TYPE(T) dt,                         \
                               const int steps);                              \
  template void Qme<T>::Evolve1(Eigen::Ref<DenseVector<T,Eigen::Dynamic>> rho, REAL_TYPE(T) dt);

DECLARE_EXPLICIT_INSTANTIATIONS(complex64);
DECLARE_EXPLICIT_INSTANTIATIONS(complex128);

}
