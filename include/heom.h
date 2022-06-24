/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef HEOM_H

#define HEOM_H

#include "qme.h"

#include "linalg_engine/linalg_engine_nil.h"

#include "hrchy_space.h"

namespace libheom
{


template<typename dtype, bool order, typename linalg_engine>
class heom : public qme_base<dtype,order,linalg_engine>
{
 public:
  using env = engine_env<linalg_engine>;
  
  hrchy_space hs;
  vector<vector<int>> lk;

  vector<dtype> ngamma_diag;
  std::unique_ptr<lil_matrix<dynamic,dtype,order,nil>[]> gamma_offdiag_lil;
  std::unique_ptr<vector<dtype>[]> s;
  std::unique_ptr<vector<dtype>[]> a;

  int max_depth;
  int n_inner_threads;
  int n_outer_threads;
  
  int n_dim;
  int n_hrchy;
  
  heom(int max_depth, int n_inner_threads, int n_outer_threads)
      : qme_base<dtype,order,linalg_engine>::qme_base()
  {
    this->max_depth = max_depth;
    this->n_inner_threads = n_inner_threads;
    this->n_outer_threads = n_outer_threads;
  }

  int get_n_hrchy()
  {
    return n_hrchy;
  }
  
  virtual void set_param(linalg_engine* obj)
  {
    CALL_TRACE();
    qme_base<dtype,order,linalg_engine>::set_param(obj);

    this->hs.n_dim
        = std::accumulate(&this->len_gamma[0], &this->len_gamma[0]+this->n_noise, 0);

    // linearlize
    this->lk.resize(this->n_noise);
    int ctr_lk = 0;
    for (int u = 0; u < this->n_noise; ++u) {
      this->lk[u].resize(this->len_gamma[u]);
      for (int k = 0; k < this->len_gamma[u]; ++k) {
        this->lk[u][k] = ctr_lk;
        ++ctr_lk;
      }
    }

    // alloc hrchy_space
    auto callback_wrapper = [&](int lidx, int estimated_max_lidx)
    {
    };
    auto filter_wrapper   = [&](std::vector<int> index, int depth) -> bool
    {
      return false;
    };
    int interval_callback = 1;
    this->n_hrchy = alloc_hrchy_space(this->hs,
                                      max_depth,
                                      callback_wrapper,
                                      interval_callback,
                                      filter_wrapper,
                                      false // filter_flag
                                      );

    // calculate ngamma_diag
    this->ngamma_diag.resize(this->n_hrchy);
    for (int lidx = 0; lidx < this->n_hrchy; ++lidx) {
      this->ngamma_diag[lidx] = zero<dtype>();
      for (int u = 0; u < this->n_noise; ++u) {
        for (int k = 0; k < this->len_gamma[u]; ++k) {
          this->ngamma_diag[lidx]
              += static_cast<real_t<dtype>>(this->hs.n[lidx][this->lk[u][k]])
              *this->gamma_lil[u].data[k][k];
        }
      }
    }

    // calculate gamma_offdiag
    this->gamma_offdiag_lil.reset(new lil_matrix<dynamic,dtype,order,nil>[this->n_noise]);
    
    for (int u = 0; u < this->n_noise; ++u) {
      this->gamma_lil[u].set_shape(this->len_gamma[u], this->len_gamma[u]);
      for (auto& gamma_ijv : this->gamma_lil[u].data) {
        int i = gamma_ijv.first;
        for (auto& gamma_jv: gamma_ijv.second) {
          int j = gamma_jv.first;
          const dtype& v = gamma_jv.second;
          if (i != j) {
            this->gamma_offdiag_lil[u].data[i][j] = v;  // TODO: should be optimized
          }
        }
      }
    }

    // calculate s and a
    this->s.reset(new vector<dtype>[this->n_noise]);
    this->a.reset(new vector<dtype>[this->n_noise]);

    for (int u = 0; u < this->n_noise; ++u) {
      this->s[u].resize(this->len_gamma[u]);
      this->a[u].resize(this->len_gamma[u]);
      gemv(nilobj,
           one<dtype>(),  this->S_lil[u], &this->phi_0[u][0],
           zero<dtype>(), &this->s[u][0],
           this->len_gamma[u]);
      gemv(nilobj,
           one<dtype>(),  this->A_lil[u], &this->phi_0[u][0],
           zero<dtype>(), &this->a[u][0],
           this->len_gamma[u]);
    }
  }
};

}

#endif
