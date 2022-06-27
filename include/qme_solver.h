/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef LIBHEOM_QME_SOLVER_H
#define LIBHEOM_QME_SOLVER_H

#include <iostream>

#include "qme_base.h"
#include "solver/solver_base.h"

namespace libheom {

template<typename dtype, order_t order, typename linalg_engine>
class qme_solver
{
 public:
  linalg_engine* engine;
  qme_base<dtype,order,linalg_engine>* qme;
  solver_base<dtype,order,linalg_engine>* solver;

  qme_solver(linalg_engine* engine_,
             qme_base<dtype,order,linalg_engine>* qme_,
             solver_base<dtype,order,linalg_engine>* solver_)
      : engine(engine_), qme(qme_), solver(solver_)
  {
    CALL_TRACE();
    int main_size  = qme->main_size();
    int temp_size  = qme->temp_size();
    solver->init(this->engine, main_size, temp_size);
  }

  void calc_diff(dtype* drho_dt,
                 dtype* rho)
  {
    CALL_TRACE();
    solver->calc_diff(qme, drho_dt, rho, 1, 0);
  }

  void solve(dtype* rho,
             const real_t<dtype>* t_list,
             int n_t,
             std::function<void(real_t<dtype>)> callback,
             const kwargs_t& kwargs)
  {
    CALL_TRACE();
    solver->solve(qme, rho, t_list, n_t, callback, kwargs);
  }
};

}

#endif
