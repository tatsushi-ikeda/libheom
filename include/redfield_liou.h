/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef REDFIELD_LIOU_H
#define REDFIELD_LIOU_H

#include "redfield.h"

namespace libheom
{

template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   order_t order_,
                   typename linalg_engine_> class matrix_base,
         order_t order,
         order_t order_liou,
         typename linalg_engine>
class redfield_liou : public redfield<dtype,order,linalg_engine>
{
 public:
  constexpr static int n_level_c_2 = n_level_c*n_level_c;
  
  using env = engine_env<linalg_engine>;
  lil_matrix<dynamic,dtype,order_liou,nil> L_lil;
  lil_matrix<dynamic,dtype,order_liou,nil> R_lil;
  
  std::unique_ptr<lil_matrix<dynamic,dtype,order_liou,nil>[]> Phi_lil;
  std::unique_ptr<lil_matrix<dynamic,dtype,order_liou,nil>[]> Theta_lil;
  
  matrix_base<n_level_c_2,dtype,order_liou,linalg_engine> R;

  redfield_liou(): redfield<dtype,order,linalg_engine>()
  {};
  
  int main_size()
  {
    CALL_TRACE();
    return this->n_level*this->n_level;
  }

  int temp_size()
  {
    CALL_TRACE();
    return 0;
  }

  void set_param(linalg_engine* obj)
  {
    CALL_TRACE();
    redfield<dtype,order,linalg_engine>::set_param(obj);
    
    this->L_lil.set_shape(this->n_level_2, this->n_level_2);
    kron_x_1  <dynamic>(nilobj, +i_unit<dtype>(), this->H_lil, zero<dtype>(), this->L_lil);
    kron_1_x_T<dynamic>(nilobj, -i_unit<dtype>(), this->H_lil,  one<dtype>(), this->L_lil);

    this->Phi_lil.reset(new lil_matrix<dynamic,dtype,order_liou,nil>[this->n_noise]);
    this->Theta_lil.reset(new lil_matrix<dynamic,dtype,order_liou,nil>[this->n_noise]);
    
    for (int s = 0; s < this->n_noise; ++s) {
      this->Phi_lil[s].set_shape(this->n_level_2, this->n_level_2);
      kron_x_1  <dynamic>(nilobj, +i_unit<dtype>(), this->V_lil[s], zero<dtype>(), this->Phi_lil[s]);
      kron_1_x_T<dynamic>(nilobj, -i_unit<dtype>(), this->V_lil[s],  one<dtype>(), this->Phi_lil[s]);

      this->Theta_lil[s].set_shape(this->n_level_2, this->n_level_2);
      kron_x_1  <dynamic>(nilobj, +i_unit<dtype>(), this->Lambda_lil[s],     zero<dtype>(), this->Theta_lil[s]);
      kron_1_x_T<dynamic>(nilobj, -i_unit<dtype>(), this->Lambda_dgr_lil[s],  one<dtype>(), this->Theta_lil[s]);
    }

    this->R_lil.set_shape(this->n_level_2, this->n_level_2);
    for (int s = 0; s < this->n_noise;
         ++s) {
      gemm<dynamic>(nilobj,
                    -one<dtype>(), this->Phi_lil[s], this->Theta_lil[s],
                    one<dtype>(), this->R_lil, this->n_level_2);
    }
    axpy<dynamic>(nilobj, one<dtype>(), this->L_lil, this->R_lil, this->n_level_2);

    this->R_lil.optimize();
    this->R.import(this->R_lil);
  }

  inline void calc_diff_impl(linalg_engine* linalg_engine_obj,
                             device_t<dtype,env>* drho_dt,
                             device_t<dtype,env>* rho,
                             dtype alpha,
                             dtype beta,
                             device_t<dtype,env>* temp)
  {
    CALL_TRACE();
    gemv<n_level_c_2>(linalg_engine_obj, -alpha, this->R, rho, beta, drho_dt, this->n_level_2);
  }
};

}

#endif
