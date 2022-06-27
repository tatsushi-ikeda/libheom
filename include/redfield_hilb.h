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
  matrix_base<n_level_c,dtype,order,linalg_engine> H;
  std::unique_ptr<matrix_base<n_level_c,dtype,order,linalg_engine>[]> V;
  std::unique_ptr<matrix_base<n_level_c,dtype,order,linalg_engine>[]> Lambda;
  std::unique_ptr<matrix_base<n_level_c,dtype,order,linalg_engine>[]> Lambda_dgr;

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
    this->H.import(this->H_lil);
    this->V.reset(new matrix_base<n_level_c,dtype,order,linalg_engine>[this->n_noise]);
    this->Lambda.reset    (new matrix_base<n_level_c,dtype,order,linalg_engine>[this->n_noise]);
    this->Lambda_dgr.reset(new matrix_base<n_level_c,dtype,order,linalg_engine>[this->n_noise]);
    for (int s = 0; s < this->n_noise; ++s) {
      this->V[s].import(this->V_lil[s]);
      this->Lambda    [s].import(this->Lambda_lil    [s]);
      this->Lambda_dgr[s].import(this->Lambda_dgr_lil[s]);
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
    gemm<n_level_c>(obj, -i_unit<dtype>()*alpha, this->H, rho, beta,         drho_dt, this->n_level);
    gemm<n_level_c>(obj,  i_unit<dtype>()*alpha, rho, this->H, one<dtype>(), drho_dt, this->n_level);
// #define OUT(x) std::cout << #x << ":" << x << std::endl;
//     std::cout << "rho" << ":" << rho[0] << "," << rho[1] << "," << rho[2] << "," << rho[3] << std::endl;
//     std::cout << "drho_dt" << ":" << drho_dt[0] << "," << drho_dt[1] << "," << drho_dt[2] << "," << drho_dt[3] << std::endl;
//     OUT(alpha);
//     OUT(beta);
//     OUT(this->H.data);
    // std::cout << "alpha:" << alpha << std::endl;
    for (int s = 0; s < this->n_noise; ++s) {
      // OUT(this->Lambda[s].data);
      // OUT(this->Lambda_dgr[s].data);
      // OUT(this->V[s].data);
      gemm<n_level_c>(obj, +i_unit<dtype>(), this->Lambda[s], rho,    zero<dtype>(), temp, this->n_level);
      gemm<n_level_c>(obj, -i_unit<dtype>(), rho, this->Lambda_dgr[s], one<dtype>(), temp, this->n_level);
      gemm<n_level_c>(obj, +i_unit<dtype>()*alpha, this->V[s], temp, one<dtype>(), drho_dt, this->n_level);
      gemm<n_level_c>(obj, -i_unit<dtype>()*alpha, temp, this->V[s], one<dtype>(), drho_dt, this->n_level);
    }
    // std::cout << "drho_dt'" << ":" << drho_dt[0] << "," << drho_dt[1] << "," << drho_dt[2] << "," << drho_dt[3] << std::endl;
  }
};

}

#endif
