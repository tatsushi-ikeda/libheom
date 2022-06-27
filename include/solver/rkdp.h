/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef RKDP_H
#define RKDP_H

#include "adaptive_step_size_solver.h"

namespace libheom {

template<typename dtype, order_t order, typename linalg_engine>
class rkdp : public adaptive_step_size_solver<dtype,order,linalg_engine>
{
 public:
  typedef engine_env<linalg_engine> env;

  const dtype A[6][6] = {
    {},
    {frac<dtype>(1,5)},
    {frac<dtype>(3,40),
     frac<dtype>(9,40)},
    {frac<dtype>(44,45),
     frac<dtype>(-56,15),
     frac<dtype>(32,9)},
    {frac<dtype>(19372,6561),
     frac<dtype>(-25360,2187),
     frac<dtype>(64448,6561),
     frac<dtype>(-212,729)},
    {frac<dtype>(9017,3168),
     frac<dtype>(-355,33),
     frac<dtype>(46732,5247),
     frac<dtype>(49,176),
     frac<dtype>(-5103,18656)},
  };

  const dtype B[6] = {
    frac<dtype>(35,384),
    zero<dtype>(),
    frac<dtype>(500,1113),
    frac<dtype>(125,192),
    frac<dtype>(-2187,6784),
    frac<dtype>(11,84)
  };

  const dtype E[7] = {
    frac<dtype>(-71,57600),
    zero<dtype>(),
    frac<dtype>(71,16695),
    frac<dtype>(-71,1920),
    frac<dtype>(17253,339200),
    frac<dtype>(-22,525),
    frac<dtype>(1,40)
  };

  const real_t<dtype> error_exponent = -frac<real_t<dtype>>(1,5);

  device_t<dtype,env>* rho_n;
  device_t<dtype,env>* rho_old;
  device_t<dtype,env>* kh[7];
  int main_size;

  void init(linalg_engine* engine,
            const int main_size,
            const int temp_size)
  {
    CALL_TRACE();
    adaptive_step_size_solver<dtype,order,linalg_engine>::init(engine, main_size, temp_size);
    this->main_size = main_size;
    this->buff_dev  = new_dev<dtype,env>(main_size*9);
    this->rho_n   = &this->buff_dev[main_size*0];
    this->rho_old = &this->buff_dev[main_size*1];
    for (int i = 0; i < 7; ++i) {
      this->kh[i]  = &this->buff_dev[main_size*(i+2)];
    }
  }

  void solve_adaptive_step(qme_base<dtype,order,linalg_engine>* qme,
                           device_t<dtype,env>* rho,
                           real_t<dtype>& t,
                           real_t<dtype> t_bound,
                           real_t<dtype>& dt_1,
                           const kwargs_t& kwargs)
  {
    CALL_TRACE();
    bool accepted = false;
    real_t<dtype> safety = 0.9;
    const real_t<dtype> dt_min = 1e-16;

    copy<dynamic>(this->engine, rho, this->rho_old, this->main_size);
    int count = 0;
    int count_max = 1000;

    while (not accepted) {
      ++count;
      if (count > count_max) {
        t = t_bound;
        break;
      }

      if (t + dt_1 > t_bound) {
        dt_1 = t_bound - t;
      }
      // std::chrono::system_clock::time_point  start = std::chrono::system_clock::now();
      // std::chrono::system_clock::time_point end   = std::chrono::system_clock::now();
      // std::chrono::duration dur_misc = end - start;
      // std::chrono::duration dur_diff = end - start;
      for (int i = 0; i < 6; ++i) {
        // start = std::chrono::system_clock::now();

        copy<dynamic>(this->engine, rho, this->rho_n, this->main_size);

        for (int j = 0; j < i; ++j) {

          axpy<dynamic>(this->engine, this->A[i][j], this->kh[j], this->rho_n, this->main_size);
        }
        // end = std::chrono::system_clock::now();
        // dur_misc += end - start;

        // start = std::chrono::system_clock::now();
        qme->calc_diff_impl(this->engine, this->kh[i], this->rho_n, dt_1,  0, this->temp_dev);
        // end = std::chrono::system_clock::now();
        // dur_diff += end - start;
      }

      // start = std::chrono::system_clock::now();
      for (int i = 0; i < 6; ++i) {
        axpy<dynamic>(this->engine, this->B[i], this->kh[i], rho, this->main_size);
      }
      // end = std::chrono::system_clock::now();
      // dur_misc += end - start;
      // start = std::chrono::system_clock::now();
      qme->calc_diff_impl(this->engine, this->kh[6], rho, dt_1,  0, this->temp_dev);
      // end = std::chrono::system_clock::now();
      // dur_diff += end - start;

      // kh[0] = error
      // start = std::chrono::system_clock::now();
      scal<dynamic>(this->engine, this->E[0], this->kh[0], this->main_size);
      for (int i = 1; i < 7; ++i) {
        axpy<dynamic>(this->engine, this->E[i], this->kh[i], this->kh[0], this->main_size);
      }
      // end = std::chrono::system_clock::now();
      // dur_misc += end - start;

      // maximum<dynamic>(this->engine, rtol, rho, this->rho_old, this->kh[1], this->main_size);
      // real_t<dtype> scale = atol + std::max(abs_old, abs) * rtol;
      // div<dynamic>(this->engine, this->kh[1], this->kh[0], this->main_size);
      // start = std::chrono::system_clock::now();
      real_t<dtype> error_norm = errnrm1<dynamic>(this->engine, this->kh[0], rho, this->rho_old, this->atol, this->rtol, this->main_size); // *dt_1
      // end = std::chrono::system_clock::now();
      // std::chrono::duration dur_norm = end - start;
      // auto msec_misc = std::chrono::duration_cast<std::chrono::milliseconds>(dur_misc).count();
      // auto msec_diff = std::chrono::duration_cast<std::chrono::milliseconds>(dur_diff).count();
      // auto msec_norm = std::chrono::duration_cast<std::chrono::milliseconds>(dur_norm).count();
      // std::cerr << "msec_misc:" << msec_misc << ",msec_diff:" << msec_diff << ",msec_norm:" << msec_norm << std::endl;
      // std::cerr << "dt" << dt_1 << ", error:" << error_norm << std::endl;
      // std::cerr << rho[0].real() << ","
      //           << rho[1].real() << ","
      //           << rho[2].real() << ","
      //           << rho[3].real()
      //           << std::endl;
      // real_t<dtype> error = std::sqrt(std::abs(dotc<dynamic>(this->engine, this->kh[0], this->kh[0], this->main_size)))*dt_1/scale;
      if (error_norm < one<real_t<dtype>>()) {
        t += dt_1;
        dt_1 *= safety*std::min(frac<real_t<dtype>>(10,1), std::pow(error_norm, error_exponent));
        accepted = true;
      } else {
        dt_1 *= safety*std::max(frac<real_t<dtype>>(1,5),  std::pow(error_norm, error_exponent));
        if (dt_1 < dt_min) {
          dt_1 = dt_min;
        }
        copy<dynamic>(this->engine, this->rho_old, rho, this->main_size);
      }
    }
  }
};

}

#endif
