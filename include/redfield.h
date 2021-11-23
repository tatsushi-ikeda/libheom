/*
 * LibHEOM, Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef REDFIELD_H
#define REDFIELD_H

#include "qme.h"

namespace libheom
{

template<typename T>
class redfield
    : public qme<T>
{
 public:
  lil_matrix<T> Z;
  std::unique_ptr<bool[]>                        use_corr_func;
  std::unique_ptr<std::function<T(real_t<T>)>[]> corr_func;
  std::unique_ptr<lil_matrix<T>[]>               Lambda;
  std::vector<T>                                 sub_vector;
  
  void init_aux_vars();
};


template<typename T,
         template <typename, int> class matrix_type,
         int num_state>
class redfield_h
    : public redfield<T>
{
 public:
  using matrix_hilb = matrix_type<T,num_state>;
  
  matrix_hilb                    H_impl;
  std::unique_ptr<matrix_hilb[]> V_impl;
  std::unique_ptr<matrix_hilb[]> Lambda_impl;
  std::unique_ptr<matrix_hilb[]> Lambda_dagger_impl;
  matrix_hilb                    X_impl;

  // std::vector<T> tmp_vector;

  void init_aux_vars
  /**/();

  void calc_diff
  /**/( ref<dense_vector<T,Eigen::Dynamic>>              drho_dt,
        const ref<const dense_vector<T,Eigen::Dynamic>>& rho,
        real_t<T> alpha,
        real_t<T> beta) override;
  
  // void ConstructCommutator(lil_matrix<T>& x,
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
         template <typename,int> class matrix_type,
         int num_state>
class redfield_l
    : public redfield<T>
{
 public:
  constexpr static int num_state_liou = n_state_prod(num_state,num_state);
  using matrix_liou = matrix_type<T,num_state_liou>;
  int n_state_liou;

  lil_matrix<T>                    L;
  std::unique_ptr<lil_matrix<T>[]> Phi;
  std::unique_ptr<lil_matrix<T>[]> Theta;
  lil_matrix<T>                    R_redfield;
  
  matrix_liou                      R_redfield_impl;
  matrix_liou                      X_impl;

  // std::vector<T> tmp_vector;

  void init_aux_vars
  /**/();

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
  
  // void ApplyCommutator(T* rho) override;
  
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}

#endif
