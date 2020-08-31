/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef HEOM_H
#define HEOM_H

#include "qme.h"
#include "hierarchy_space.h"

namespace libheom {

// Heom{H,L}{D,S}{L,H}

template<typename T>
class Heom
  : public Qme<T> {
 public:
  HierarchySpace hs;
  std::vector<T> jgamma_diag;
  std::unique_ptr<Eigen::SparseMatrix<T, Eigen::RowMajor>[]> gamma_offdiag;
  
  std::unique_ptr<Eigen::Matrix<T,Eigen::Dynamic,1>[]> S;
  std::unique_ptr<Eigen::Matrix<T,Eigen::Dynamic,1>[]> A;
  
  int n_dim;
  int n_hierarchy;

  void LinearizeDim();
  
  void Initialize();
  void InitAuxVars(std::function<void(int)> callback);
};


template<typename T>
class HeomL
   : public Heom<T> {
 public:
  int n_state_liou;
  // Liouville space operators
  LilMatrix<T> L;
  std::unique_ptr<LilMatrix<T>[]> Phi;
  std::unique_ptr<LilMatrix<T>[]> Psi;
  std::unique_ptr<std::unique_ptr<LilMatrix<T>[]>[]> Theta;
  std::unique_ptr<LilMatrix<T>[]> Xi;
  LilMatrix<T> R_heom_0;
  
  LilMatrix<T> X;

  void InitAuxVars(std::function<void(int)> callback);
};


template<typename T,
         template <typename, int> class MatrixType,
         int NumState>
class HeomLL
    : public HeomL<T> {
 public:
  constexpr static int NumStateLiou = n_state_prod(NumState,NumState);
  using MatrixLiou = MatrixType<T,NumStateLiou>;
  
  // Auxiliary variables
  MatrixLiou L_impl;
  std::unique_ptr<MatrixLiou[]> Phi_impl;
  std::unique_ptr<MatrixLiou[]> Psi_impl;
  std::unique_ptr<MatrixLiou[]> Xi_impl;
  MatrixLiou R_heom_0_impl;

  // std::vector<T> sub_vector;

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
  
  // void ApplyCommutator(Ref<DenseVector<T>>& rho) override;
  
  void InitAuxVars(std::function<void(int)> callback);

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


template<typename T,
         template <typename, int> class MatrixType,
         int NumState>
class HeomLH
    : public HeomL<T> {
 public:
  // Hierarchical Liouville space operators
  LilMatrix<T> R_heom;
  LilMatrix<T> X_hrchy;

  // Auxiliary variables
  MatrixType<T,Eigen::Dynamic> R_heom_impl;
  MatrixType<T,Eigen::Dynamic> X_hrchy_impl;

  std::vector<T> sub_vector;

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
  
  // void ApplyCommutator(Ref<DenseVector<T>>& rho) override;
  
  void InitAuxVars(std::function<void(int)> callback);

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


}
#endif /* HEOM_H */
