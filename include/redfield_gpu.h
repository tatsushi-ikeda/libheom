/*
 * LibHEOM, Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef REDFIELD_GPU_H
#define REDFIELD_GPU_H

#include <memory>

#include "redfield.h"

namespace libheom {

template<typename T, template <typename, int> class matrix_type, int num_state>
class redfield_h_gpu_vars;

template<typename T, template <typename, int> class matrix_type, int num_state>
class redfield_h_gpu
    : public redfield_h<T, matrix_type, num_state> {
public:
  void calc_diff(ref<dense_vector<T,Eigen::Dynamic>> drho_dt,
                 const ref<const dense_vector<T,Eigen::Dynamic>>& rho,
                 REAL_TYPE(T) alpha,
                 REAL_TYPE(T) beta) override;

  void evolve(ref<dense_vector<T,Eigen::Dynamic>> rho,
              REAL_TYPE(T) dt,
              const int steps);

  void evolve_1(ref<dense_vector<T,Eigen::Dynamic>> rho,
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
  
  void set_device_number(int device_number);
  
  void init_aux_vars(std::function<void(int)> callback);

  int device_number = -1;

  std::shared_ptr<redfield_h_gpu_vars<T, matrix_type, num_state>> gpu;
};

}

#endif
