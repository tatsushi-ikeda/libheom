/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef ADAPTIVE_STEP_SIZE_SOLVER_H
#define ADAPTIVE_STEP_SIZE_SOLVER_H

#include "solver_base.h"

namespace libheom {

template<typename dtype, bool order, typename linalg_engine>
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
               real_t<dtype> t_0,
               real_t<dtype> t_1,
               const kwargs_t& kwargs)
  {
    CALL_TRACE();
    if (this->dt_save == -1) {
      this->dt_save = std::any_cast<real_t<dtype>>(kwargs.at("dt"));
    }
    this->atol    = std::any_cast<real_t<dtype>>(kwargs.at("atol"));
    this->rtol    = std::any_cast<real_t<dtype>>(kwargs.at("rtol"));

    real_t<dtype> t = t_0;
    real_t<dtype> dt_1 = this->dt_save;

    while (true) {
      solve_adaptive_step(qme, rho, t, t_1, dt_1, kwargs);
      if (t >= t_1) {
        break;
      }
    }

    this->dt_save = dt_1;
  }

  virtual void solve_adaptive_step(qme_base<dtype,order,linalg_engine>* qme,
                                   device_t<dtype,env>* rho,
                                   real_t<dtype>& t,
                                   real_t<dtype> t_bound,
                                   real_t<dtype>& dt_1,
                                   const kwargs_t& kwargs)
  {
    CALL_TRACE();
  }
};

}

#endif
