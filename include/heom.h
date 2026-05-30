/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LICENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef LIBHEOM_HEOM_H
#define LIBHEOM_HEOM_H

#include "qme_base.h"

#include "linalg_engine/linalg_engine_nil.h"

#include "hierarchy_space.h"

namespace libheom
{

template<typename dtype, order_t order, typename linalg_engine>
class heom : public qme_base<dtype,order,linalg_engine>
{
 public:
  using env = engine_env<linalg_engine>;

  hierarchy_space hs;
  vector<vector<int>> lk;

  vector<dtype> n_gamma_diag;
  std::unique_ptr<lil_matrix<dynamic,dtype,order,nil>[]> gamma_offdiag;
  std::unique_ptr<vector<dtype>[]> s_vec;
  std::unique_ptr<vector<dtype>[]> a_vec;

  int truncation_depth;
  int n_inner_threads;
  int n_outer_threads;

  int n_modes;
  int n_hierarchy;

  heom() = delete;

  heom(int truncation_depth, int n_inner_threads, int n_outer_threads)
      : qme_base<dtype,order,linalg_engine>::qme_base()
  {
    this->truncation_depth = truncation_depth;
    this->n_inner_threads = n_inner_threads;
    this->n_outer_threads = n_outer_threads;
  }

  int get_n_hierarchy()
  {
    return n_hierarchy;
  }

  virtual void set_param(linalg_engine* obj)
  {
    CALL_TRACE();
    qme_base<dtype,order,linalg_engine>::set_param(obj);

    this->hs.n_modes
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

    // alloc hierarchy_space
    this->n_hierarchy = alloc_hierarchy_space(this->hs, truncation_depth);

    // calculate n_gamma_diag
    this->n_gamma_diag.resize(this->n_hierarchy);
    for (int lidx = 0; lidx < this->n_hierarchy; ++lidx) {
      this->n_gamma_diag[lidx] = zero<dtype>();
      for (int u = 0; u < this->n_noise; ++u) {
        for (int k = 0; k < this->len_gamma[u]; ++k) {
          this->n_gamma_diag[lidx]
              += static_cast<real_t<dtype>>(this->hs.n[lidx][this->lk[u][k]])
              *this->gamma[u].data[k][k];
        }
      }
    }

    // calculate gamma_offdiag
    this->gamma_offdiag.reset(new lil_matrix<dynamic,dtype,order,nil>[this->n_noise]);

    for (int u = 0; u < this->n_noise; ++u) {
      this->gamma[u].set_shape(this->len_gamma[u], this->len_gamma[u]);
      for (auto& gamma_ijv : this->gamma[u].data) {
        int i = gamma_ijv.first;
        for (auto& gamma_jv : gamma_ijv.second) {
          int j = gamma_jv.first;
          const dtype& v = gamma_jv.second;
          if (i != j) {
            // Pre-filter off-diagonal entries so calc_time_derivative can iterate
            // gamma_offdiag without an i != j branch inside the n_hierarchy loop.
            this->gamma_offdiag[u].data[i][j] = v;
          }
        }
      }
    }

    // project s_mat/a_mat onto phi_0 to get per-mode scalar vectors s_vec/a_vec
    this->s_vec.reset(new vector<dtype>[this->n_noise]);
    this->a_vec.reset(new vector<dtype>[this->n_noise]);

    for (int u = 0; u < this->n_noise; ++u) {
      this->s_vec[u].resize(this->len_gamma[u]);
      this->a_vec[u].resize(this->len_gamma[u]);
      gemv(nilobj,
           one<dtype>(),  this->s_mat[u], &this->phi_0[u][0],
           zero<dtype>(), &this->s_vec[u][0],
           this->len_gamma[u]);
      gemv(nilobj,
           one<dtype>(),  this->a_mat[u], &this->phi_0[u][0],
           zero<dtype>(), &this->a_vec[u][0],
           this->len_gamma[u]);
    }
  }
};

}

#endif
