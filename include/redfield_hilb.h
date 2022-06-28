/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef LIBHEOM_REDFIELD_HILB_H
#define LIBHEOM_REDFIELD_HILB_H

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
         typename linalg_engine>
class redfield_hilb : public redfield<dtype,order,linalg_engine>
{
 public:
  using env = engine_env<linalg_engine>;

  struct {
    matrix_base<n_level_c,dtype,order,linalg_engine> H;
    std::unique_ptr<matrix_base<n_level_c,dtype,order,linalg_engine>[]> V;
    std::unique_ptr<matrix_base<n_level_c,dtype,order,linalg_engine>[]> Lambda;
    std::unique_ptr<matrix_base<n_level_c,dtype,order,linalg_engine>[]> Lambda_dgr;
  } impl;

  redfield_hilb(): redfield<dtype,order,linalg_engine>()
  {};

  int main_size()
  {
    CALL_TRACE();
    return this->n_level*this->n_level;
  }

  int temp_size()
  {
    CALL_TRACE();
    return this->n_level*this->n_level;
  }

  virtual void set_param(linalg_engine* obj)
  {
    CALL_TRACE();
    redfield<dtype,order,linalg_engine>::set_param(obj);
    this->impl.H.import(this->H);
    this->impl.V.reset(new matrix_base<n_level_c,dtype,order,linalg_engine>[this->n_noise]);
    this->impl.Lambda.reset    (new matrix_base<n_level_c,dtype,order,linalg_engine>[this->n_noise]);
    this->impl.Lambda_dgr.reset(new matrix_base<n_level_c,dtype,order,linalg_engine>[this->n_noise]);
    for (int s = 0; s < this->n_noise; ++s) {
      this->impl.V[s].import(this->V[s]);
      this->impl.Lambda    [s].import(this->Lambda    [s]);
      this->impl.Lambda_dgr[s].import(this->Lambda_dgr[s]);
    }
  }

  inline void calc_diff_impl(linalg_engine* obj,
                             device_t<dtype,env>* drho_dt,
                             device_t<dtype,env>* rho,
                             dtype alpha,
                             dtype beta,
                             device_t<dtype,env>* temp)
  {
    CALL_TRACE();
    auto n_level = this->n_level;
    auto n_noise = this->n_noise;
    auto& H = this->impl.H;
    auto& V = this->impl.V;
    auto& Lambda = this->impl.Lambda;
    auto& Lambda_dgr = this->impl.Lambda_dgr;
    
    gemm<n_level_c>(obj, -i_unit<dtype>()*alpha, H, rho, beta,         drho_dt, n_level);
    gemm<n_level_c>(obj,  i_unit<dtype>()*alpha, rho, H, one<dtype>(), drho_dt, n_level);
    for (int s = 0; s < n_noise; ++s) {
      gemm<n_level_c>(obj, +i_unit<dtype>(), Lambda[s], rho,    zero<dtype>(), temp, n_level);
      gemm<n_level_c>(obj, -i_unit<dtype>(), rho, Lambda_dgr[s], one<dtype>(), temp, n_level);
      gemm<n_level_c>(obj, +i_unit<dtype>()*alpha, V[s], temp, one<dtype>(), drho_dt, n_level);
      gemm<n_level_c>(obj, -i_unit<dtype>()*alpha, temp, V[s], one<dtype>(), drho_dt, n_level);
    }
  }
};

}

#endif
