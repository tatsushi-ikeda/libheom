/*
 * LibHEOM: Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef HEOM_H
#define HEOM_H

#include "qme.h"
#include "hrchy_space.h"

namespace libheom {

// heom{h,l}{d,s}{l,h}

template<typename T>
class heom
    : public qme<T>
{
 public:
  hrchy_space hs;
  std::vector<T>  ngamma_diag;
  std::unique_ptr<Eigen::SparseMatrix<T, Eigen::RowMajor>[]> gamma_offdiag;
  
  std::unique_ptr<Eigen::Matrix<T,Eigen::Dynamic,1>[]> S;
  std::unique_ptr<Eigen::Matrix<T,Eigen::Dynamic,1>[]> A;
  
  int n_dim;
  int n_hrchy;
  
  void linearize
  /**/();
  
  void init
  /**/();
  
  void init_aux_vars
  /**/();
  
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


template<typename T>
class heom_l
    : public heom<T>
{
 public:
  int n_state_liou;

  // Liouville space operators
  lil_matrix<T> L;
  std::unique_ptr<lil_matrix<T>[]> Phi;
  std::unique_ptr<lil_matrix<T>[]> Psi;
  std::unique_ptr<std::unique_ptr<lil_matrix<T>[]>[]> Theta;
  std::unique_ptr<lil_matrix<T>[]> Xi;
  lil_matrix<T> R_heom_0;
  
  lil_matrix<T> X;

  void init_aux_vars
  /**/();

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


template<typename T,
         template <typename, int> class matrix_type,
         int num_state>
class heom_ll
    : public heom_l<T>
{
 public:
  constexpr static int num_state_liou = n_state_prod(num_state,num_state);
  using matrix_liou = matrix_type<T,num_state_liou>;
  
  // Auxiliary variables
  matrix_liou L_impl;
  std::unique_ptr<matrix_liou[]> Phi_impl;
  std::unique_ptr<matrix_liou[]> Psi_impl;
  std::unique_ptr<matrix_liou[]> Xi_impl;
  matrix_liou R_heom_0_impl;

  // std::vector<T> sub_vector;
  void calc_diff
  /**/(ref<dense_vector<T,Eigen::Dynamic>> drho_dt,
       const ref<const dense_vector<T,Eigen::Dynamic>>& rho,
       real_t<T> alpha,
       real_t<T> beta) override;

  // void ConstructCommutator(LilMatrix<T>& x,
  //                          T coef_l,
  //                          T coef_r,
  //                          std::function<void(int)> callback
  //                          = [](int) { return; },
  //                          int interval_callback = 1024) override;
  
  // void ApplyCommutator(ref<dense_vector<T>>& rho) override;
  
  void init_aux_vars
  /**/();

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


template<typename T,
         template <typename, int> class matrix_type,
         int num_state>
class heom_lh
    : public heom_l<T>
{
 public:
  // Hierarchical Liouville space operators
  lil_matrix<T> R_heom;
  lil_matrix<T> X_hrchy;

  // Auxiliary variables
  matrix_type<T,Eigen::Dynamic> R_heom_impl;
  matrix_type<T,Eigen::Dynamic> X_hrchy_impl;

  std::vector<T> sub_vector;

  void calc_diff
  /**/(ref<dense_vector<T,Eigen::Dynamic>> drho_dt,
       const ref<const dense_vector<T,Eigen::Dynamic>>& rho,
       real_t<T> alpha,
       real_t<T> beta) override;

  // void ConstructCommutator(lil_matrix<T>& x,
  //                          T coef_l,
  //                          T coef_r,
  //                          std::function<void(int)> callback
  //                          = [](int) { return; },
  //                          int interval_callback = 1024) override;
  
  // void ApplyCommutator(ref<dense_vector<T>>& rho) override;
  
  void init_aux_vars
  /**/();

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


}
#endif /* HEOM_H */
