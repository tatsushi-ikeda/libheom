/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef LIBHEOM_REDFIELD_H
#define LIBHEOM_REDFIELD_H

#include "qme_base.h"

#include "linalg_engine/linalg_engine_nil.h"

namespace libheom
{


template<typename dtype, order_t order, typename linalg_engine>
class redfield : public qme_base<dtype,order,linalg_engine>
{
 public:
  using env = engine_env<linalg_engine>;
  using qme_base<dtype,order,linalg_engine>::qme_base;

  std::unique_ptr<lil_matrix<dynamic,dtype,order,nil>[]> I;
  std::unique_ptr<lil_matrix<dynamic,dtype,order,nil>[]> C;
  std::unique_ptr<vector<dtype>[]> sigma_T_C;
  
  std::unique_ptr<lil_matrix<dynamic,dtype,order,nil>[]> Lambda;
  std::unique_ptr<lil_matrix<dynamic,dtype,order,nil>[]> Lambda_dgr;

  inline dtype correlation(int s, real_t<dtype> omega)
  {
    lil_matrix<dynamic,dtype,order,nil> A(this->gamma[s]);
    vector<dtype> x(this->len_gamma[s]);
    axpy(nilobj, -i_unit<dtype>()*omega, I[s], A, this->len_gamma[s]);
    lu_solve(A, this->phi_0[s], x);
    // TODO: remove gevm from implemention.
    return dotu<dynamic>(nilobj, &this->sigma_T_C[s][0], &x[0], this->len_gamma[s]) + this->s_delta[s];
  }

  virtual void set_param(linalg_engine* obj)
  {
    CALL_TRACE();
    qme_base<dtype,order,linalg_engine>::set_param(obj);
    
    // this->gamma.reset(new dense_matrix<dynamic,dtype,order,eigen>[this->n_noise]);
    // this->S.reset(new dense_matrix<dynamic,dtype,order,eigen>[this->n_noise]);
    // this->A.reset(new dense_matrix<dynamic,dtype,order,eigen>[this->n_noise]);
    // for (int s = 0; s < this->n_noise; ++s) {
    //   this->gamma[s].import(this->gamma[s]);
    //   this->S[s].import(this->S[s]);
    //   this->A[s].import(this->A[s]);
    // }
    
    this->I.reset(new lil_matrix<dynamic,dtype,order,nil>[this->n_noise]);
    for (int s = 0; s < this->n_noise; ++s) {
      this->I[s].set_identity(this->len_gamma[s]);
    }

    this->C.reset(new lil_matrix<dynamic,dtype,order,nil>[this->n_noise]);
    for (int s = 0; s < this->n_noise; ++s) {
      this->C[s] = this->S[s];
      axpy(nilobj, i_unit<dtype>(), this->A[s], this->C[s], this->len_gamma[s]);
    }

    this->sigma_T_C.reset(new vector<dtype>[this->n_level]);
    for (int s = 0; s < this->n_noise; ++s) {
      sigma_T_C[s].resize(this->len_gamma[s]);
      gevm(nilobj,
           one<dtype>(), &this->sigma[s][0], this->C[s],
           zero<dtype>(), &this->sigma_T_C[s][0],
           this->len_gamma[s]);
    }
    
    this->Lambda    .reset(new lil_matrix<dynamic,dtype,order,nil>[this->n_noise]);
    this->Lambda_dgr.reset(new lil_matrix<dynamic,dtype,order,nil>[this->n_noise]);

    dense_matrix<dynamic,dtype,order,linalg_engine> H_tmp;
    dense_matrix<dynamic,dtype,order,linalg_engine> V_tmp;
    dense_matrix<dynamic,dtype,order,linalg_engine> Lambda_tmp;
    H_tmp.import(this->H);
    
    vector<real_t<dtype>> w(this->n_level);
    device_t<real_t<dtype>,env>* w_dev = new_dev<real_t<dtype>,env,true>(this->n_level);
    
    device_t<dtype,env>* v_dev = new_dev<dtype,env>(this->n_level*this->n_level);

    host2dev<real_t<dtype>,env>(&w[0], w_dev, this->n_level);
    eig<dynamic>(obj, H_tmp, w_dev, v_dev, this->n_level);
    dev2host<real_t<dtype>,env>(w_dev, &w[0], this->n_level);

    vector<dtype> Lambda_v(this->n_level*this->n_level);
    device_t<dtype,env>* Lambda_v_dev = new_dev<dtype,env,true>(this->n_level*this->n_level);
    
    vector<dtype> V_v(this->n_level*this->n_level);
    device_t<dtype,env>* V_v_dev = new_dev<dtype,env,true>(this->n_level*this->n_level);

    // lil_matrix<dynamic,dtype,col_major,nil> A(3,3);
    // vector<dtype> b(3);

    // // A.data[0][0] = 1;
    // // A.data[0][1] = 1;
    // // A.data[0][2] = 3;

    // // A.data[1][0] = 1;
    // // A.data[1][1] = 8;
    // // A.data[1][2] = 1;

    // // A.data[2][0] = 1;
    // // A.data[2][1] = 1;
    // // A.data[2][2] = -1;

    // A.data[0][0] = 1;
    // A.data[1][0] = 1;
    // A.data[2][0] = 3;

    // A.data[0][1] = 1;
    // A.data[1][1] = 8;
    // A.data[2][1] = 1;

    // A.data[0][2] = 1;
    // A.data[1][2] = 1;
    // A.data[2][2] = -1;
    
    // b[0] = 0;
    // b[1] = 4;
    // b[2] = -4;
    

    // std::cout << A << std::endl;
    
    // vector<dtype> x(3);
    // lu_solve(A, b, x);
    
    // std::cout << x[0] << "," << std::endl;
    // std::cout << x[1] << "," << std::endl;
    // std::cout << x[2] << "," << std::endl;

    // std::cout << "checked!" << std::endl;
    // std::exit(1);
    
    for (int s = 0; s < this->n_noise; ++s) {
      V_tmp.import(this->V[s]);
      host2dev<dtype,env>(&V_v[0], V_v_dev, this->n_level*this->n_level);
      utf<dynamic>(obj, V_tmp, v_dev, V_v_dev, this->n_level);
      dev2host<dtype,env>(V_v_dev, &V_v[0], this->n_level*this->n_level);
      for (int i = 0; i < this->n_level; ++i) {
        for (int j = 0; j < this->n_level; ++j) {
          dtype val = V_v[i*this->n_level + j];
          real_t<dtype> omega_ji;
          if constexpr (order == row_major) {
            omega_ji = std::real(w[j] - w[i]);
          } else {
            omega_ji = std::real(w[i] - w[j]);
          }
          dtype corr = correlation(s, omega_ji);
          Lambda_v[i*this->n_level+j] = val*corr;
        }
      }
      Lambda_tmp.set_shape(this->n_level, this->n_level);
      host2dev<dtype,env>(&Lambda_v[0], Lambda_v_dev, this->n_level*this->n_level);
      utb<dynamic>(obj, Lambda_v_dev, v_dev, Lambda_tmp, this->n_level);
      dev2host<dtype,env>(Lambda_v_dev, &Lambda_v[0], this->n_level*this->n_level);
      Lambda_tmp.dump(this->Lambda[s]);
      this->Lambda_dgr[s].set_adjoint(this->Lambda[s]);
    }

    delete_dev<real_t<dtype>,env,true>(w_dev);
    delete_dev<dtype,env>(v_dev);
    delete_dev<dtype,env,true>(Lambda_v_dev);
    delete_dev<dtype,env,true>(V_v_dev);
  }
};

}

#endif
