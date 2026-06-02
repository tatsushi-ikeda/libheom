/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LICENSE.txt for license.
 *------------------------------------------------------------------------*/

#ifndef LIBHEOM_ADAPTIVE_STEP_SIZE_SOLVER_H
#define LIBHEOM_ADAPTIVE_STEP_SIZE_SOLVER_H

#include "solver_base.h"

namespace libheom {

template<typename dtype, order_t order, typename linalg_engine>
class adaptive_step_size_solver : public solver_base<dtype,order,linalg_engine>
{
 public:
  typedef engine_env<linalg_engine> env;

  real_t<dtype> dt_save, atol, rtol;

  virtual void init(linalg_engine* engine,
                    const int main_size,
                    const int temp_size)
  {
    CALL_TRACE();
    solver_base<dtype,order,linalg_engine>::init(engine, main_size, temp_size);
    this->dt_save = -1;
  }

  void solve_1(qme_base<dtype,order,linalg_engine>* qme,
               device_t<dtype,env>* rho,
               real_t<dtype> t_start,
               real_t<dtype> t_end,
               const kwargs_t& kwargs)
  {
    CALL_TRACE();
    if (this->dt_save == -1) {
      this->dt_save = get_kwarg<real_t<dtype>>(kwargs, "dt");
    }
    this->atol = get_kwarg<real_t<dtype>>(kwargs, "atol");
    this->rtol = get_kwarg<real_t<dtype>>(kwargs, "rtol");

    real_t<dtype> t  = t_start;
    real_t<dtype> dt = this->dt_save;

    while (true) {
      solve_adaptive_step(qme, rho, t, t_end, dt, kwargs);
      if (t >= t_end) {
        break;
      }
    }

    this->dt_save = dt;
  }

  virtual void solve_adaptive_step(qme_base<dtype,order,linalg_engine>* qme,
                                   device_t<dtype,env>* rho,
                                   real_t<dtype>& t,
                                   real_t<dtype> t_bound,
                                   real_t<dtype>& dt,
                                   const kwargs_t& kwargs)
  {
    CALL_TRACE();
  }
};

}

#endif
