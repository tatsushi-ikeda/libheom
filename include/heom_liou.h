/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef HEOM_LIOU_H

#define HEOM_LIOU_H

#include "heom.h"
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
class heom_liou : public heom<dtype,order,linalg_engine>
{
 public:
  constexpr static int n_level_c_2 = n_level_c*n_level_c;
  int count;
  using env = engine_env<linalg_engine>;
  using heom<dtype,order,linalg_engine>::heom;

  lil_matrix<dynamic,dtype,order_liou,nil> L_lil;
  lil_matrix<dynamic,dtype,order_liou,nil> R_0_lil;
  std::unique_ptr<lil_matrix<dynamic,dtype,order_liou,nil>[]> Phi_lil;
  std::unique_ptr<lil_matrix<dynamic,dtype,order_liou,nil>[]> Psi_lil;
  std::unique_ptr<lil_matrix<dynamic,dtype,order_liou,nil>[]> Xi_lil;

  matrix_base<n_level_c_2,dtype,order_liou,linalg_engine> R_0;
  std::unique_ptr<matrix_base<n_level_c_2,dtype,order_liou,linalg_engine>[]> Phi;
  std::unique_ptr<matrix_base<n_level_c_2,dtype,order_liou,linalg_engine>[]> Psi;
  
  heom_liou(): heom<dtype,order,linalg_engine>()
  {
    count = 0;
  };
  
  int main_size()
  {
    CALL_TRACE();
    return this->n_level*this->n_level*this->n_hrchy;
  }

  int temp_size()
  {
    CALL_TRACE();
    return this->n_level*this->n_level*3*this->n_outer_threads;
  }
  
  virtual void set_param(linalg_engine* obj)
  {
    CALL_TRACE();
    heom<dtype,order,linalg_engine>::set_param(obj);

    this->L_lil.set_shape(this->n_level_2, this->n_level_2);
    kron_x_1  <dynamic>(nilobj, +i_unit<dtype>(), this->H_lil, zero<dtype>(), this->L_lil);
    kron_1_x_T<dynamic>(nilobj, -i_unit<dtype>(), this->H_lil,  one<dtype>(), this->L_lil);
    
    this->Phi_lil.reset(new lil_matrix<dynamic,dtype,order_liou,nil>[this->n_noise]);
    this->Psi_lil.reset(new lil_matrix<dynamic,dtype,order_liou,nil>[this->n_noise]);
    this->Xi_lil.reset (new lil_matrix<dynamic,dtype,order_liou,nil>[this->n_noise]);
    
    for (int u = 0; u < this->n_noise; ++u) {
      this->Phi_lil[u].set_shape(this->n_level_2, this->n_level_2);
      kron_x_1  <dynamic>(nilobj, +i_unit<dtype>(), this->V_lil[u], zero<dtype>(), this->Phi_lil[u]);
      kron_1_x_T<dynamic>(nilobj, -i_unit<dtype>(), this->V_lil[u],  one<dtype>(), this->Phi_lil[u]);
      this->Phi_lil[u].optimize();
      
      this->Psi_lil[u].set_shape(this->n_level_2, this->n_level_2);
      kron_x_1  <dynamic>(nilobj, +one<dtype>(), this->V_lil[u], zero<dtype>(), this->Psi_lil[u]);
      kron_1_x_T<dynamic>(nilobj, +one<dtype>(), this->V_lil[u],  one<dtype>(), this->Psi_lil[u]);
      this->Psi_lil[u].optimize();
      
      this->Xi_lil[u].set_shape(this->n_level_2, this->n_level_2);
      gemm<dynamic>(nilobj, -this->s_delta[u], this->Phi_lil[u], this->Phi_lil[u], zero<dtype>(), this->Xi_lil[u], this->n_level_2);
      this->Xi_lil[u].optimize();
    }

    this->R_0_lil.set_shape(this->n_level_2, this->n_level_2);
    axpy<dynamic>(nilobj, one<dtype>(), this->L_lil, this->R_0_lil, this->n_level_2);
    for (int u = 0; u < this->n_noise; ++u) {
      axpy<dynamic>(nilobj, one<dtype>(), this->Xi_lil[u], this->R_0_lil, this->n_level_2);
    }

    this->R_0.import(this->R_0_lil);
    this->Phi.reset(new matrix_base<n_level_c_2,dtype,order_liou,linalg_engine>[this->n_noise]);
    this->Psi.reset(new matrix_base<n_level_c_2,dtype,order_liou,linalg_engine>[this->n_noise]);
    for (int u = 0; u < this->n_noise; ++u) {
      this->Phi[u].import(this->Phi_lil[u]);
      this->Psi[u].import(this->Psi_lil[u]);
    }

    for (int u = 0; u < this->n_noise; ++u) {
      auto& gamma_offdiag_u_lil = this->gamma_offdiag_lil[u];
      for (auto& gamma_jkv : gamma_offdiag_u_lil.data) {
        int j = gamma_jkv.first;
        for (auto& gamma_kv: gamma_jkv.second) {
          int k = gamma_kv.first;
          const dtype& v = gamma_kv.second;
          std::cout << j << ", " << k << ": " << v << std::endl;
        }
      }
    }
  }

  inline void calc_diff_impl(linalg_engine* obj_base,
                             device_t<dtype,env>* drho_dt,
                             device_t<dtype,env>* rho,
                             dtype alpha,
                             dtype beta,
                             device_t<dtype,env>* temp_base)
  {
    CALL_TRACE();
    ++this->count;
    
    auto n_hrchy        = this->n_hrchy;
    auto n_level        = this->n_level;
    auto n_level_2      = this->n_level_2;
    auto n_noise        = this->n_noise;
    auto& ngamma_diag   = this->ngamma_diag;
    auto& n             = this->hs.n;
    auto& ptr_m1        = this->hs.ptr_m1;
    auto& ptr_p1        = this->hs.ptr_p1;
    const auto ptr_void = this->hs.ptr_void;

    obj_base->set_n_inner_threads(this->n_inner_threads);
    obj_base->set_n_outer_threads(this->n_outer_threads);
    omp_set_max_active_levels(2);

    obj_base->create_children(this->n_outer_threads);
#pragma omp parallel for num_threads(this->n_outer_threads)
    for (int lidx = 0; lidx < n_hrchy; ++lidx) {
      int thread_id = omp_get_thread_num();
      linalg_engine* obj = static_cast<linalg_engine*>(obj_base->get_child(thread_id));
      obj->switch_thread(thread_id);
      
      auto rho_n     = &rho[lidx*n_level_2];

      auto drho_dt_n = &temp_base[(3*thread_id+0)*n_level_2];
      auto temp_Phi  = &temp_base[(3*thread_id+1)*n_level_2];
      auto temp_Psi  = &temp_base[(3*thread_id+2)*n_level_2];
      
      // 0 terms
      gemv<n_level_c_2>(obj, one<dtype>(), this->R_0, rho_n, zero<dtype>(), drho_dt_n, n_level_2);
      axpy<n_level_c_2>(obj, this->ngamma_diag[lidx], rho_n, drho_dt_n, n_level_2);

      for (int u = 0; u < n_noise; ++u) {
        auto& lk_u                = this->lk[u];
        auto  len_gamma_u         = this->len_gamma[u];
        auto& gamma_offdiag_u_lil = this->gamma_offdiag_lil[u];
        auto& sigma_u             = this->sigma[u];
        auto& s_u                 = this->s[u];
        auto& a_u                 = this->a[u];

        
        for (auto& gamma_jkv : gamma_offdiag_u_lil.data) {
          int j = gamma_jkv.first;
          for (auto& gamma_kv: gamma_jkv.second) {
            int k = gamma_kv.first;
            const dtype& v = gamma_kv.second;
            if constexpr (order == row_major) {
              int lidx_m1j    = ptr_m1[lidx][lk_u[j]];
              int lidx_m1jp1k = ptr_p1[lidx_m1j][lk_u[k]];
            
              if (lidx_m1j != ptr_void && lidx_m1jp1k != ptr_void) {
                auto rho_m1jp1k = &rho[lidx_m1jp1k*n_level_2];
                auto n_j_float = static_cast<real_t<dtype>>(n[lidx][lk_u[j]]);
#ifdef LIBHEOM_SQRT_NORMALIZATION            
                auto n_k_float = static_cast<real_t<dtype>>(n[lidx][lk_u[k]]);
                
                axpy<n_level_c_2>(obj, std::sqrt(n_j_float*(n_k_float + 1))*v, rho_m1jp1k, drho_dt_n, n_level_2);
#else            
                axpy<n_level_c_2>(obj, n_j_float*v, rho_m1jp1k, drho_dt_n, n_level_2);
#endif
              }
            } else {
              int lidx_m1k    = ptr_m1[lidx][lk_u[k]];
              int lidx_m1kp1j = ptr_p1[lidx_m1k][lk_u[j]];
            
              if (lidx_m1k != ptr_void && lidx_m1kp1j != ptr_void) {
                auto rho_m1kp1j = &rho[lidx_m1kp1j*n_level_2];
                auto n_k_float = static_cast<real_t<dtype>>(n[lidx][lk_u[k]]);
#ifdef LIBHEOM_SQRT_NORMALIZATION            
                auto n_j_float = static_cast<real_t<dtype>>(n[lidx][lk_u[j]]);
              
                axpy<n_level_c_2>(obj, std::sqrt(n_k_float*(n_j_float + 1))*v, rho_m1kp1j, drho_dt_n, n_level_2);
#else            
                axpy<n_level_c_2>(obj, n_k_float*v, rho_m1kp1j, drho_dt_n, n_level_2);
#endif
              }
            }
          }
        }
        
        nullify<n_level_c_2>(obj, temp_Phi, n_level_2);
        nullify<n_level_c_2>(obj, temp_Psi, n_level_2);
        
        // +1 terms
        for (int k = 0; k < len_gamma_u; ++k) {
          int lidx_p1 = ptr_p1[lidx][lk_u[k]];
          if (lidx_p1 != ptr_void) {
            auto rho_np1 = &rho[lidx_p1*n_level_2];
#ifdef LIBHEOM_SQRT_NORMALIZATION            
            auto n_float = static_cast<real_t<dtype>>(n[lidx][lk_u[k]]);
            axpy<n_level_c_2>(obj, sigma_u[k]*std::sqrt(n_float + 1), rho_np1, temp_Phi, n_level_2);
#else        
            axpy<n_level_c_2>(obj, sigma_u[k],                        rho_np1, temp_Phi, n_level_2);
#endif
          }
        }

        // -1 terms
        for (int k = 0; k < len_gamma_u; ++k) {
          int lidx_m1 = ptr_m1[lidx][lk_u[k]];
          auto rho_nm1 = &rho[lidx_m1*n_level_2];
          if (lidx_m1 != ptr_void) {
            auto n_float = static_cast<real_t<dtype>>(n[lidx][lk_u[k]]);
#ifdef LIBHEOM_SQRT_NORMALIZATION
            axpy<n_level_c_2>(obj,  s_u[k]*std::sqrt(n_float), rho_nm1, temp_Phi, n_level_2);
            axpy<n_level_c_2>(obj, -a_u[k]*std::sqrt(n_float), rho_nm1, temp_Psi, n_level_2);
#else        
            axpy<n_level_c_2>(obj,  s_u[k]*n_float,            rho_nm1, temp_Phi, n_level_2);
            axpy<n_level_c_2>(obj, -a_u[k]*n_float,            rho_nm1, temp_Psi, n_level_2);
#endif
          }
        }
        
        gemv<n_level_c_2>(obj,  one<dtype>(), this->Phi[u], temp_Phi, one<dtype>(), drho_dt_n, n_level_2);
        gemv<n_level_c_2>(obj,  one<dtype>(), this->Psi[u], temp_Psi, one<dtype>(), drho_dt_n, n_level_2);

        
      }
      scal<n_level_c_2>(obj, beta, &drho_dt[lidx*n_level_2], n_level_2);
      axpy<n_level_c_2>(obj, -alpha, drho_dt_n, &drho_dt[lidx*n_level_2], n_level_2);
    }
    obj_base->set_n_inner_threads(-1);
    obj_base->set_n_outer_threads(-1);
  }
};

}

#endif
