/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef RK4_H
#define RK4_H

#include "fixed_step_size_solver.h"

namespace libheom {

template<typename dtype, bool order, typename linalg_engine>
class rk4 : public fixed_step_size_solver<dtype,order,linalg_engine>
{
 public:
  typedef engine_env<linalg_engine> env;
  device_t<dtype,env>* rho_n;
  device_t<dtype,env>* kh_1;
  device_t<dtype,env>* kh_2;
  device_t<dtype,env>* kh_3;
  int main_size;

  void init(linalg_engine* engine,
            const int main_size,
            const int temp_size)
  {
    CALL_TRACE();
    fixed_step_size_solver<dtype,order,linalg_engine>::init(engine, main_size, temp_size);
    this->main_size = main_size;
    this->buff_dev = new_dev<dtype,env>(main_size*4);
    this->rho_n = &this->buff_dev[main_size*0];
    this->kh_1  = &this->buff_dev[main_size*1];
    this->kh_2  = &this->buff_dev[main_size*2];
    this->kh_3  = &this->buff_dev[main_size*3];
  }

  void solve_fixed_step(qme_base<dtype,order,linalg_engine>* qme,
                        device_t<dtype,env>* rho,
                        real_t<dtype> t,
                        real_t<dtype> dt_1,
                        const kwargs_t& kwargs)
  {
    CALL_TRACE();
    copy<dynamic>(this->engine, rho, this->rho_n, this->main_size);
    qme->calc_diff_impl(this->engine, this->kh_1, this->rho_n, dt_1,  0, this->temp_dev);

    axpy<dynamic>(this->engine, frac<dtype>(1,2), this->kh_1, this->rho_n, this->main_size);
    qme->calc_diff_impl(this->engine, this->kh_2, this->rho_n, dt_1,  0, this->temp_dev);

    copy<dynamic>(this->engine, rho, this->rho_n, this->main_size);
    axpy<dynamic>(this->engine, frac<dtype>(1,2), this->kh_2, this->rho_n, this->main_size);
    qme->calc_diff_impl(this->engine, this->kh_3, this->rho_n, dt_1,  0, this->temp_dev);

    copy<dynamic>(this->engine, rho, this->rho_n, this->main_size);
    axpy<dynamic>(this->engine, one<dtype>(), this->kh_3, this->rho_n, this->main_size);
    axpy<dynamic>(this->engine, one<dtype>(), this->kh_3, this->kh_2,  this->main_size);
    qme->calc_diff_impl(this->engine, this->kh_3, this->rho_n, dt_1,  0, this->temp_dev);

    axpy<dynamic>(this->engine, frac<dtype>(1,6), this->kh_1, rho, this->main_size);
    axpy<dynamic>(this->engine, frac<dtype>(2,6), this->kh_2, rho, this->main_size);
    axpy<dynamic>(this->engine, frac<dtype>(1,6), this->kh_3, rho, this->main_size);
  }
};

}

#endif
