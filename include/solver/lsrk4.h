/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef LSRK4_H
#define LSRK4_H

#include "fixed_step_size_solver.h"

namespace libheom {

template<typename dtype, order_t order, typename linalg_engine>
class lsrk4 : public fixed_step_size_solver<dtype,order,linalg_engine>
{
 public:
  int main_size;
  typedef engine_env<linalg_engine> env;

  void init(linalg_engine* engine,
            const int main_size,
            const int temp_size)
  {
    CALL_TRACE();
    fixed_step_size_solver<dtype,order,linalg_engine>::init(engine, main_size, temp_size);
    this->main_size = main_size;
    this->buff_dev = new_dev<dtype,env>(main_size);
  }

  void solve_fixed_step(qme_base<dtype,order,linalg_engine>* qme,
                        device_t<dtype,env>* rho,
                        real_t<dtype> t,
                        real_t<dtype> dt_1,
                        const kwargs_t& kwargs)
  {
    CALL_TRACE();
    qme->calc_diff_impl(this->engine, this->buff_dev, rho, dt_1,  0, this->temp_dev);
    axpy<dynamic>(this->engine, frac<dtype>(1,3), this->buff_dev, rho, this->main_size);
    qme->calc_diff_impl(this->engine, this->buff_dev, rho, dt_1, -1, this->temp_dev);
    axpy<dynamic>(this->engine, frac<dtype>(3,4), this->buff_dev, rho, this->main_size);
    qme->calc_diff_impl(this->engine, this->buff_dev, rho, dt_1, -1, this->temp_dev);
    axpy<dynamic>(this->engine, frac<dtype>(2,3), this->buff_dev, rho, this->main_size);
    qme->calc_diff_impl(this->engine, this->buff_dev, rho, dt_1, -1, this->temp_dev);
    axpy<dynamic>(this->engine, frac<dtype>(1,4), this->buff_dev, rho, this->main_size);
  }
};

}

#endif
