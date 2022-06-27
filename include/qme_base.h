/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef LIBHEOM_QME_BASE_H
#define LIBHEOM_QME_BASE_H

#include "type.h"
#include "const.h"

#include "env.h"

#include "linalg_engine/matrix_base.h"
#include "linalg_engine/lil_matrix.h"
#include "linalg_engine/linalg_engine.h"

namespace libheom
{

template<typename dtype, order_t order, typename linalg_engine>
class qme_base
{
 public:
  using env  = engine_env<linalg_engine>;

  qme_base(): n_level(0)
  {};
  
  int n_level;
  int n_level_2;
  
  lil_matrix<dynamic,dtype,order,nil> H_lil;

  int n_noise;
  
  std::unique_ptr<lil_matrix<dynamic,dtype,order,nil>[]> V_lil;

  std::unique_ptr<int[]> len_gamma;
  std::unique_ptr<lil_matrix<dynamic,dtype,order,nil>[]> gamma_lil;
  std::unique_ptr<vector<dtype>[]> phi_0;
  std::unique_ptr<vector<dtype>[]> sigma;
  
  std::unique_ptr<lil_matrix<dynamic,dtype,order,nil>[]> S_lil;
  std::unique_ptr<lil_matrix<dynamic,dtype,order,nil>[]> A_lil;
  std::unique_ptr<dtype[]> s_delta;
  
  virtual int main_size() { return 0; };

  virtual int temp_size() { return 0; };

  void alloc_noises(int n_noise)
  {
    this->n_noise = n_noise;
  
    this->V_lil.reset(new lil_matrix<dynamic,dtype,order,nil>[n_noise]);
    this->len_gamma.reset(new int [n_noise]);
    this->gamma_lil.reset(new lil_matrix<dynamic,dtype,order,nil>[n_noise]);
    this->phi_0.reset(new vector<dtype>[n_noise]);
    this->sigma.reset(new vector<dtype>[n_noise]);
  
    this->S_lil.reset(new lil_matrix<dynamic,dtype,order,nil>[n_noise]);
    this->s_delta.reset(new dtype [n_noise]);
    this->A_lil.reset(new lil_matrix<dynamic,dtype,order,nil>[n_noise]);
  }

  virtual void set_param(linalg_engine* obj)
  {
    CALL_TRACE();
  }

  virtual inline void calc_diff_impl(linalg_engine* linalg_engine_obj,
                                     device_t<dtype,env>* drho_dt,
                                     device_t<dtype,env>* rho,
                                     dtype alpha,
                                     dtype beta,
                                     device_t<dtype,env>* temp) {};
};

}

#endif
