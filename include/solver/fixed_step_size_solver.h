/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef FIXED_STEP_SIZE_SOLVER_H
#define FIXED_STEP_SIZE_SOLVER_H

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
               real_t<dtype> t_0,
               real_t<dtype> t_1,
               const kwargs_t& kwargs)
  {
    CALL_TRACE();
    real_t<dtype> t = t_0;
    real_t<dtype> dt_1 = std::any_cast<real_t<dtype>>(kwargs.at("dt"));
    bool break_flag = false;
    while (true) {
      if (t + dt_1 - t_1 > -std::numeric_limits<real_t<dtype>>::epsilon()) {
        dt_1 = t_1 - t;
        break_flag = true;
      }

      solve_fixed_step(qme, rho_dev, t, dt_1, kwargs);

      t += dt_1;
      if (break_flag) {
        break;
      }
    }
  }

  virtual void solve_fixed_step(qme_base<dtype,order,linalg_engine>* qme,
                                device_t<dtype,env>* rho,
                                real_t<dtype> t,
                                real_t<dtype> dt_1,
                                const kwargs_t& kwargs)
  {
    CALL_TRACE();
  }
};

}

#endif
