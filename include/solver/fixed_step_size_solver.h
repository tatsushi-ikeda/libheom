/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LICENSE.txt for license.
 *------------------------------------------------------------------------*/

#ifndef LIBHEOM_FIXED_STEP_SIZE_SOLVER_H
#define LIBHEOM_FIXED_STEP_SIZE_SOLVER_H

#include "solver_base.h"

namespace libheom {

template<typename dtype, order_t order, typename linalg_engine>
class fixed_step_size_solver : public solver_base<dtype,order,linalg_engine>
{
 public:
  typedef engine_env<linalg_engine> env;

  virtual void init(linalg_engine* engine,
                    const int main_size,
                    const int temp_size)
  {
    CALL_TRACE();
    solver_base<dtype,order,linalg_engine>::init(engine, main_size, temp_size);
  }

  void solve_1(qme_base<dtype,order,linalg_engine>* qme,
               device_t<dtype,env>* rho_dev,
               real_t<dtype> t_start,
               real_t<dtype> t_end,
               const kwargs_t& kwargs)
  {
    CALL_TRACE();
    real_t<dtype> t  = t_start;
    real_t<dtype> dt = get_kwarg<real_t<dtype>>(kwargs, "dt");
    bool break_flag = false;
    while (true) {
      if (t + dt - t_end > -std::numeric_limits<real_t<dtype>>::epsilon()) {
        dt = t_end - t;
        break_flag = true;
      }

      solve_fixed_step(qme, rho_dev, t, dt, kwargs);

      t += dt;
      if (break_flag) {
        break;
      }
    }
  }

  virtual void solve_fixed_step(qme_base<dtype,order,linalg_engine>* qme,
                                device_t<dtype,env>* rho,
                                real_t<dtype> t,
                                real_t<dtype> dt,
                                const kwargs_t& kwargs)
  {
    CALL_TRACE();
  }
};

}

#endif
