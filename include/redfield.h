/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef REDFIELD_H
#define REDFIELD_H

#include "qme.h"

namespace libheom {

template<typename T>
class Redfield
    : public Qme<T> {
 public:
  LilMatrix<T> Z;
  std::unique_ptr<bool[]> use_corr_func;
  std::unique_ptr<std::function<T(REAL_TYPE(T))>[]> corr_func;
  std::unique_ptr<LilMatrix<T>[]> Lambda;
  std::vector<T> sub_vector;  
  
  void InitAuxVars(std::function<void(int)> callback);
};


template<typename T,
         template <typename, int> class MatrixType,
         int NumState>
class RedfieldH
    : public Redfield<T> {
 public:
  using MatrixHilb = MatrixType<T,NumState>;
  
  MatrixHilb H_impl;
  std::unique_ptr<MatrixHilb[]> V_impl;
  std::unique_ptr<MatrixHilb[]> Lambda_impl;
  std::unique_ptr<MatrixHilb[]> Lambda_dagger_impl;
  MatrixHilb X_impl;

  // std::vector<T> tmp_vector;

  void InitAuxVars(std::function<void(int)> callback);

  void CalcDiff(Ref<DenseVector<T,Eigen::Dynamic>> drho_dt,
                const Ref<const DenseVector<T,Eigen::Dynamic>>& rho,
                REAL_TYPE(T) alpha,
                REAL_TYPE(T) beta) override;
  
  // void ConstructCommutator(LilMatrix<T>& x,
  //                          T coef_l,
  //                          T coef_r,
  //                          std::function<void(int)> callback
  //                          = [](int) { return; },
  //                          int interval_callback = 1024) override;
  
  // void ApplyCommutator(T* rho) override;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


template<typename T,
         template <typename,int> class MatrixType,
         int NumState>
class RedfieldL
    : public Redfield<T> {
 public:
  constexpr static int NumStateLiou = n_state_prod(NumState,NumState);
  using MatrixLiou = MatrixType<T,NumStateLiou>;
  int n_state_liou;

  LilMatrix<T> L;
  std::unique_ptr<LilMatrix<T>[]> Phi;
  std::unique_ptr<LilMatrix<T>[]> Theta;
  LilMatrix<T> R_redfield;
  
  MatrixLiou R_redfield_impl;
  MatrixLiou X_impl;

  // std::vector<T> tmp_vector;

  void InitAuxVars(std::function<void(int)> callback);

  void CalcDiff(Ref<DenseVector<T,Eigen::Dynamic>> drho_dt,
                const Ref<const DenseVector<T,Eigen::Dynamic>>& rho,
                REAL_TYPE(T) alpha,
                REAL_TYPE(T) beta) override;
  
  // void ConstructCommutator(LilMatrix<T>& x,
  //                          T coef_l,
  //                          T coef_r,
  //                          std::function<void(int)> callback
  //                          = [](int) { return; },
  //                          int interval_callback = 1024) override;
  
  // void ApplyCommutator(T* rho) override;
  
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


}

#endif
