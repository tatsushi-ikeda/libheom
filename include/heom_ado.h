/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef HEOM_ADO_H

#define HEOM_ADO_H

#include "heom_liou.h"
#include <omp.h>

namespace libheom
{


template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   bool order_,
                   typename linalg_engine_> class matrix_base,
         bool order,
         bool order_liou,
         typename linalg_engine>
class heom_ado : public heom_liou<n_level_c,dtype,matrix_base,order,order_liou,linalg_engine>
{
 public:
  constexpr static int n_level_c_2 = n_level_c*n_level_c;
  
  using env = engine_env<linalg_engine>;
  using heom_liou<n_level_c,dtype,matrix_base,order,order_liou,linalg_engine>::heom_liou;
  int n_level_ado;
  
  std::unique_ptr<std::unique_ptr<lil_matrix<dynamic,dtype,order_liou,nil>[]>[]> Theta_lil;

  lil_matrix<dynamic,dtype,order_liou,nil> R_lil;
  matrix_base<dynamic,dtype,order_liou,linalg_engine> R;
  
  heom_ado(): heom_liou<n_level_c,dtype,matrix_base,order,order_liou,linalg_engine>()
  {};
  
  int main_size()
  {
    CALL_TRACE();
    return this->n_level*this->n_level*this->n_hrchy;
  }

  int temp_size()
  {
    CALL_TRACE();
    return 0;
  }
  
  virtual void set_param(linalg_engine* obj)
  {
    CALL_TRACE();
    heom_liou<n_level_c,dtype,matrix_base,order,order_liou,linalg_engine>::set_param(obj);

    this->Theta_lil.reset(new std::unique_ptr<lil_matrix<dynamic,dtype,order_liou,nil>[]>[this->n_noise]);
    
    for (int u = 0; u < this->n_noise; ++u) {
      this->Theta_lil[u].reset(new lil_matrix<dynamic,dtype,order_liou,nil>[this->len_gamma[u]]);
      for (int k = 0; k < this->len_gamma[u]; ++k) {
        this->Theta_lil[u][k].set_shape(this->n_level_2, this->n_level_2);
        axpy<dynamic>(nilobj, +this->s[u][k], this->Phi_lil[u], this->Theta_lil[u][k], this->n_level_2);
        axpy<dynamic>(nilobj, -this->a[u][k], this->Psi_lil[u], this->Theta_lil[u][k], this->n_level_2);
        this->Theta_lil[u][k].optimize();
      }
    }

    this->n_level_ado = this->n_hrchy*this->n_level_2;
    this->R_lil.set_shape(n_level_ado, n_level_ado);
    
    for (int lidx = 0; lidx < this->n_hrchy; ++lidx) {
      for (int a = 0; a < this->n_level_2; ++a) {
        // -1 terms
        for (int u = 0; u < this->n_noise; ++u) {
          for (int k = 0; k < this->len_gamma[u]; ++k) {
            int lidx_m1 = this->hs.ptr_m1[lidx][this->lk[u][k]];
            if (lidx_m1 == this->hs.ptr_void) continue;
            try {
              for (auto& Theta_kv: this->Theta_lil[u][k].data[a]) {
                int b = Theta_kv.first;
                dtype v = Theta_kv.second;
#ifdef LIBHEOM_SQRT_NORMALIZATION            
                v *= std::sqrt(static_cast<real_t<dtype>>(this->hs.n[lidx][this->lk[u][k]]));
#else              
                v *= static_cast<real_t<dtype>>(this->hs.n[lidx][this->lk[u][k]]);
#endif
                if (v != zero<dtype>()) {
                  if constexpr (order_liou == row_major) {
                    this->R_lil.push(lidx*this->n_level_2 + a,
                                     lidx_m1*this->n_level_2 + b,
                                     v);
                  } else {
                    this->R_lil.push(lidx*this->n_level_2 + b,
                                     lidx_m1*this->n_level_2 + a,
                                     v);
                  }
                }
              }
            } catch (std::out_of_range&) {}
          }
        }
      
        // 0 terms
        this->R_lil.push(lidx*this->n_level_2 + a,
                         lidx*this->n_level_2 + a,
                         this->ngamma_diag[lidx]);
      
        for (int u = 0; u < this->n_noise; ++u) {
          for (auto& gamma_jkv : this->gamma_offdiag_lil[u].data) {
            int j = gamma_jkv.first;
            for (auto& gamma_kv: gamma_jkv.second) {
              int k = gamma_kv.first;
              const dtype& v = gamma_kv.second;
              int lidx_m1j, lidx_m1jp1k;
              if ((lidx_m1j = this->hs.ptr_m1[lidx][this->lk[u][j]])
                  != this->hs.ptr_void
                  && (lidx_m1jp1k = this->hs.ptr_p1[lidx_m1j][this->lk[u][k]])
                  != this->hs.ptr_void)  {
                auto n_j_float = static_cast<real_t<dtype>>(this->hs.n[lidx][this->lk[u][j]]);
#ifdef LIBHEOM_SQRT_NORMALIZATION            
                auto n_k_float = static_cast<real_t<dtype>>(this->hs.n[lidx][this->lk[u][k]]);
                this->R_lil.push(lidx*this->n_level_2 + a,
                                 lidx_m1jp1k*this->n_level_2 + a,
                                 v*std::sqrt(n_j_float*(n_k_float + 1)));
#else
                this->R_lil.push(lidx*this->n_level_2 + a,
                                 lidx_m1jp1k*this->n_level_2 + a,
                                 v*n_j_float);
#endif              
              }
            }
          }
        }
      
        try {
          for (auto& R_kv: this->R_0_lil.data[a]) {
            int b = R_kv.first;
            dtype v = R_kv.second;
            this->R_lil.push(lidx*this->n_level_2 + a,
                             lidx*this->n_level_2 + b,
                             v);
          }
        } catch (std::out_of_range&) {
        }

        // +1 terms
        for (int u = 0; u < this->n_noise; ++u) {
          for (int k = 0; k < this->len_gamma[u]; ++k) {
            int lidx_p1 = this->hs.ptr_p1[lidx][this->lk[u][k]];
            if (lidx_p1 == this->hs.ptr_void) continue;
            try {
              for (auto& Phi_kv: this->Phi_lil[u].data[a]) {
                int b = Phi_kv.first;
                dtype v = Phi_kv.second;
#ifdef LIBHEOM_SQRT_NORMALIZATION            
                v *= std::sqrt(static_cast<real_t<dtype>>(this->hs.n[lidx][this->lk[u][k]]+1));
#endif              
                v *= this->sigma[u][k];
                if (v != zero<dtype>()) {
                  if constexpr (order_liou == row_major) {
                    this->R_lil.push(lidx*this->n_level_2 + a,
                                     lidx_p1*this->n_level_2 + b,
                                     v);
                  } else {
                    this->R_lil.push(lidx*this->n_level_2 + b,
                                     lidx_p1*this->n_level_2 + a,
                                     v);
                  }
                }
              }
            } catch (std::out_of_range&) {}
          }
        }
      }
    }

    this->R.import(this->R_lil);
  }

  inline void calc_diff_impl(linalg_engine* obj,
                             device_t<dtype,env>* drho_dt,
                             device_t<dtype,env>* rho,
                             dtype alpha,
                             dtype beta,
                             device_t<dtype,env>* temp_base)
  {
    CALL_TRACE();

    gemv<dynamic>(obj, -alpha, this->R, rho, beta, drho_dt, this->n_level_ado);
  }
};

}

#endif
