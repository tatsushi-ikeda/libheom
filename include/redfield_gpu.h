/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef REDFIELD_GPU_H
#define REDFIELD_GPU_H

#include <memory>

#include "redfield.h"

namespace libheom {

template<typename T, template <typename, int> class MatrixType, int NumState>
class RedfieldHGpuVars;

template<typename T, template <typename, int> class MatrixType, int NumState>
class RedfieldHGpu
    : public RedfieldH<T, MatrixType, NumState> {
public:
  void CalcDiff(Ref<DenseVector<T,Eigen::Dynamic>> drho_dt,
                const Ref<const DenseVector<T,Eigen::Dynamic>>& rho,
                REAL_TYPE(T) alpha,
                REAL_TYPE(T) beta) override;

  void Evolve(Ref<DenseVector<T,Eigen::Dynamic>> rho,
              REAL_TYPE(T) dt,
              const int steps);

  void Evolve1(Ref<DenseVector<T,Eigen::Dynamic>> rho,
               REAL_TYPE(T) dt);
  
  // void construct_commutator
  // /**/(did_matrix<T>& x,
  //      T coef_l,
  //      T coef_r,
  //      std::function<void(int)> callback
  //      = [](int) { return; },
  //      int interval_callback = 1024);
  
  // void apply_commutator
  // /**/(T* rho);
  
  void SetDeviceNumber(int device_number);
  
  void InitAuxVars(std::function<void(int)> callback);

  int device_number = -1;

  std::shared_ptr<RedfieldHGpuVars<T, MatrixType, NumState>> gpu;
};

}

#endif
