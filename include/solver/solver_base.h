/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef SOLVER_BASE_H
#define SOLVER_BASE_H

#include "utility.h"

#include "linalg_engine/linalg_engine.h"
#include "qme.h"

#include <functional>

namespace libheom {

template<typename dtype, bool order, typename linalg_engine>
class solver_base
{
 public:
  typedef engine_env<linalg_engine> env;

  linalg_engine* engine;

  int main_size;
  device_t<dtype,env>* rho_dev;
  device_t<dtype,env>* temp_dev;
  device_t<dtype,env>* buff_dev;

  solver_base()
      : rho_dev (nullptr),
        temp_dev(nullptr),
        buff_dev(nullptr)
  {
    CALL_TRACE();
  }

  ~solver_base()
  {
    CALL_TRACE();
    if (this->rho_dev  != nullptr) {
      delete_dev<dtype,env,true>(this->rho_dev);
    }
    if (this->temp_dev != nullptr) {
      delete_dev<dtype,env>(this->temp_dev);
    }
    if (this->buff_dev != nullptr) {
      delete_dev<dtype,env>(this->buff_dev);
    }
  }

  void calc_diff(qme_base<dtype,order,linalg_engine>* qme,
                 dtype* drho_dt,
                 dtype* rho,
                 real_t<dtype> alpha,
                 real_t<dtype> beta)
  {
    CALL_TRACE();
  }

  void solve(qme_base<dtype,order,linalg_engine>* qme,
             dtype* rho,
             const real_t<dtype>* t_list,
             int n_t,
             std::function<void(real_t<dtype>)> callback,
             const kwargs_t& kwargs)
  {
    CALL_TRACE();
    for (int i = 0; i < n_t - 1; ++i) {
      callback(t_list[i]);
      host2dev<dtype,env>(rho, rho_dev, this->main_size);
      solve_1(qme, rho_dev, t_list[i], t_list[i+1], kwargs);
      dev2host<dtype,env>(rho_dev, rho, this->main_size);
    }
    callback(t_list[n_t-1]);
  }

  virtual void init(linalg_engine* engine,
                    const int main_size,
                    const int temp_size)
  {
    CALL_TRACE();
    this->engine = engine;
    // this->env_obj   = env_obj;
    this->main_size  = main_size;
    this->rho_dev    = new_dev<dtype,env,true>(main_size);
    this->temp_dev   = new_dev<dtype,env>(temp_size);
  }

  virtual void solve_1(qme_base<dtype,order,linalg_engine>* qme,
                       device_t<dtype,env>* rho,
                       real_t<dtype> t_0,
                       real_t<dtype> t_1,
                       const kwargs_t& kwargs)
  {
    CALL_TRACE();
  };

};

}

#endif
