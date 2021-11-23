/*
 * LibHEOM: Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#include "heom.h"

#include <numeric>

#include "const.h"
// #include "mkl_wrapper.h"

// Note: In order to avoid compile errors caused by the two phase name
//       lookup rule regarding template class, all class variables in
//       this source have this-> modifier.
namespace libheom {

template<typename T>
void heom<T>::linearize() {
  this->hs.n_dim
      = std::accumulate(this->len_gamma.get(),
                        this->len_gamma.get() + this->n_noise, 0);
  this->lk.resize(this->n_noise);
  
  int ctr_lk = 0;
  for (int u = 0; u < this->n_noise; ++u) {
    this->lk[u].resize(this->len_gamma[u]);
    for (int k = 0; k < this->len_gamma[u]; ++k) {
      this->lk[u][k] = ctr_lk;
      ++ctr_lk;
    }
  }
}


template<typename T>
void heom<T>::init() {
  this->size_rho = (this->n_hrchy+1)*this->n_state*this->n_state;
  this->sub_vector.resize(this->size_rho);
  this->sub_vector.fill(zero<T>());
}


template<typename T>
void heom<T>::init_aux_vars() {
  qme<T>::init_aux_vars();

  this->jgamma_diag.resize(this->n_hrchy);
  for (int lidx = 0; lidx < this->n_hrchy; ++lidx) {
    this->jgamma_diag[lidx] = zero<T>();
    for (int u = 0; u < this->n_noise; ++u) {
      for (int k = 0; k < this->len_gamma[u]; ++k) {
        this->jgamma_diag[lidx]
            += static_cast<REAL_TYPE(T)>(this->hs.j[lidx][this->lk[u][k]])
            *this->gamma[u].coeff(k,k);
      }
    }
  }

  this->gamma_offdiag.reset(new Eigen::SparseMatrix<T, Eigen::RowMajor>[this->n_noise]);
  for (int u = 0; u < this->n_noise; ++u) {
    std::vector<Eigen::Triplet<T>> list;
    for (int jj = 0; jj < this->gamma[u].rows(); ++jj) {
      for (int ptr = this->gamma[u].outerIndexPtr()[jj];
           ptr < this->gamma[u].outerIndexPtr()[jj+1];
           ++ptr) {
        int k = this->gamma[u].innerIndexPtr()[ptr];
        const T& val = this->gamma[u].valuePtr()[ptr];
        if (jj != k) {
          list.push_back(Eigen::Triplet<T>(jj,k,val));
        }
      }
    }
    this->gamma_offdiag[u].resize(this->gamma[u].rows(), this->gamma[u].cols());
    this->gamma_offdiag[u].setFromTriplets(list.begin(), list.end());
    this->gamma_offdiag[u].makeCompressed();
  }

  this->S.reset(new Eigen::Matrix<T,Eigen::Dynamic,1>[this->n_noise]);
  this->A.reset(new Eigen::Matrix<T,Eigen::Dynamic,1>[this->n_noise]);

  for (int u = 0; u < this->n_noise; ++u) {
    this->S[u].resize(this->s[u].rows());
    this->S[u].noalias() = this->s[u]*this->phi_0[u];
    this->A[u].resize(this->a[u].rows());
    this->A[u].noalias() = this->a[u]*this->phi_0[u];
  }
}

template<typename T>
void heom_l<T>::init_aux_vars() {
  heom<T>::init_aux_vars();
  
  this->n_state_liou = this->n_state*this->n_state;
  
  this->L.set_shape(this->n_state_liou, this->n_state_liou);
  kron_identity_right(+i_unit<T>(), this->H, zero<T>(), this->L);
  kron_identity_left (-i_unit<T>(), this->H, one<T>(),  this->L);
  this->L.optimize();
  
  this->Phi.reset(new lil_matrix<T> [this->n_noise]);
  this->Psi.reset(new lil_matrix<T> [this->n_noise]);
  this->Theta.reset(new std::unique_ptr<lil_matrix<T>[]> [this->n_noise]);
  this->Xi.reset(new lil_matrix<T> [this->n_noise]);
  
  for (int u = 0; u < this->n_noise; ++u) {
    this->Phi[u].set_shape(this->n_state_liou, this->n_state_liou);
    kron_identity_right(+i_unit<T>(), this->V[u], zero<T>(), this->Phi[u]);
    kron_identity_left (-i_unit<T>(), this->V[u], one<T>(),  this->Phi[u]);
    this->Phi[u].optimize();
    
    this->Psi[u].set_shape(this->n_state_liou, this->n_state_liou);
    kron_identity_right(frac<T>(1,1), this->V[u], zero<T>(), this->Psi[u]);
    kron_identity_left (frac<T>(1,1), this->V[u], one<T>(),  this->Psi[u]);
    this->Psi[u].optimize();

    this->Theta[u].reset(new lil_matrix<T>[this->len_gamma[u]]);
    for (int k = 0; k < this->len_gamma[u]; ++k) {
      this->Theta[u][k].set_shape(this->n_state_liou, this->n_state_liou);
      axpy(+this->S[u].coeff(k), this->Phi[u], this->Theta[u][k]);
      axpy(-this->A[u].coeff(k), this->Psi[u], this->Theta[u][k]);
      this->Theta[u][k].optimize();
    }

    this->Xi[u].set_shape(this->n_state_liou, this->n_state_liou);
    gemm(-this->S_delta[u], this->Phi[u], this->Phi[u], zero<T>(), this->Xi[u]);
    this->Xi[u].optimize();
  }

  this->R_heom_0.set_shape(this->n_state_liou, this->n_state_liou);
  axpy(one<T>(), this->L, this->R_heom_0);
  for (int u = 0; u < this->n_noise; ++u) {
    axpy(one<T>(), this->Xi[u], this->R_heom_0);
  }
  // this->L.optimize();

  // std::ofstream os("tmp2.dat");

  // os << "L:" << this->L << std::endl;
  // for (int u = 0; u < this->n_noise; ++u) {
  //   os << "Phi:" << this->Phi[u] << std::endl;
  //   os << "Psi:" << this->Psi[u] << std::endl;
  //   os << "Xi:" << this->Xi[u] << std::endl;
  //   os << "gamma:" << this->gamma[u] << std::endl;
  //   os << "sigma:" << this->sigma[u] << std::endl;
  //   os << "phi_0:" << this->phi_0[u] << std::endl;
  //   os << "s:" << this->s[u] << std::endl;
  //   os << "a:" << this->a[u] << std::endl;
  // }
}


//========================================================================
// HEOM Module by using Liouville space expression
//========================================================================


template<typename T,
         template<typename, int> class matrix_type,
         int num_state>
void heom_ll<T, matrix_type, num_state>::init_aux_vars() {
  heom_l<T>::init_aux_vars();

  this->L.template dump<num_state_liou>(this->L_impl);

  // this->L_impl= static_cast<matrix_type<T>>(this->L);

  this->Phi_impl.reset(new matrix_liou [this->n_noise]);
  this->Psi_impl.reset(new matrix_liou [this->n_noise]);
  this->Xi_impl.reset(new matrix_liou [this->n_noise]);

  for (int u = 0; u < this->n_noise; ++u) {
    this->Phi[u].template dump<num_state_liou>(this->Phi_impl[u]);
    this->Psi[u].template dump<num_state_liou>(this->Psi_impl[u]);
    this->Xi[u].template dump<num_state_liou>(this->Xi_impl[u]);
  }
  this->R_heom_0.template dump<num_state_liou>(this->R_heom_0_impl);
}


template<typename T,
         template<typename, int> class matrix_type,
         int num_state>
void heom_ll<T, matrix_type, num_state>::calc_diff(
    ref<dense_vector<T,Eigen::Dynamic>> drho_dt,
    const ref<const dense_vector<T,Eigen::Dynamic>>& rho,
    REAL_TYPE(T) alpha,
    REAL_TYPE(T) beta) {
  auto n_hrchy    = this->n_hrchy;
  auto n_state_liou   = this->n_state_liou;
  auto n_noise        = this->n_noise;
  auto& R_heom_0_impl = this->R_heom_0_impl;
  auto& jgamma_diag   = this->jgamma_diag;
  auto& len_gamma     = this->len_gamma;
  auto& lk            = this->lk;
  auto& j             = this->hs.j;
  auto& ptr_m1        = this->hs.ptr_m1;
  auto& ptr_p1        = this->hs.ptr_p1;
  auto& S             = this->S;
  auto& A             = this->A;
  auto& gamma_offdiag = this->gamma_offdiag;
  auto& sigma         = this->sigma;
  const auto ptr_void = this->hs.ptr_void;

  dense_vector<T,num_state_liou> tmp(this->n_state_liou);
  dense_vector<T,num_state_liou> tmp_Phi(this->n_state_liou);
  dense_vector<T,num_state_liou> tmp_Psi(this->n_state_liou);

  for (int lidx = 0; lidx < n_hrchy; ++lidx) {
    // auto rho_n     = rho.block(lidx*n_state_liou,0,n_state_liou,1);
    // auto drho_dt_n = drho_dt.block(lidx*n_state_liou,0,n_state_liou,1);
    auto rho_n     = block<num_state_liou,1>::value(rho, lidx*n_state_liou,0,n_state_liou,1);
    auto drho_dt_n = block<num_state_liou,1>::value(drho_dt, lidx*n_state_liou,0,n_state_liou,1);

    // 0 terms
    // drho_dt_n      = beta*drho_dt_n;
    tmp.noalias()  = R_heom_0_impl*rho_n;
    tmp.noalias() += jgamma_diag[lidx]*rho_n;
    
    for (int u = 0; u < n_noise; ++u) {
      auto& lk_u    = lk[u];
      auto  len_gamma_u = len_gamma[u];
      auto& gamma_offdiag_u = gamma_offdiag[u];
      auto& sigma_u = sigma[u];
      auto& S_u = S[u];
      auto& A_u = A[u];

      for (int jj = 0; jj < gamma_offdiag_u.rows(); ++jj) {
        for (int ptr = gamma_offdiag_u.outerIndexPtr()[jj];
             ptr < gamma_offdiag_u.outerIndexPtr()[jj+1]; ++ptr) {
          int k = gamma_offdiag_u.innerIndexPtr()[ptr];
          const T& val = gamma_offdiag_u.valuePtr()[ptr];
          int lidx_m1j, lidx_m1jp1k;
          if ((lidx_m1j = ptr_m1[lidx][lk_u[jj]]) != ptr_void
              && (lidx_m1jp1k = ptr_p1[lidx_m1j][lk_u[k]]) != ptr_void)  {
            auto rho_m1jp1k = block<num_state_liou,1>::value(
                rho, lidx_m1jp1k*n_state_liou,0,n_state_liou,1);
            auto k_float = static_cast<REAL_TYPE(T)>(j[lidx][lk_u[k]]);
            auto j_float = static_cast<REAL_TYPE(T)>(j[lidx][lk_u[jj]]);
            tmp.noalias() += val*std::sqrt(j_float*(k_float + 1))*rho_m1jp1k;
          }
        }
      } 
      
      auto& Phi_impl_u = this->Phi_impl[u];
      auto& Psi_impl_u = this->Psi_impl[u];
      
      tmp_Phi.setZero();
      tmp_Psi.setZero();
      
      // +1 terms
      for (int k = 0; k < len_gamma_u; ++k) {
        int lidx_p1 = ptr_p1[lidx][lk_u[k]];
        auto rho_np1 = block<num_state_liou,1>::value(rho, lidx_p1*n_state_liou,0,n_state_liou,1);
        auto j_float = static_cast<REAL_TYPE(T)>(j[lidx][lk_u[k]]);
        tmp_Phi.noalias() += sigma_u.coeff(k)*std::sqrt(j_float + 1)*rho_np1;
      }

      // -1 terms
      for (int k = 0; k < len_gamma_u; ++k) {
        int lidx_m1 = ptr_m1[lidx][lk_u[k]];
        auto rho_nm1 = block<num_state_liou,1>::value(rho, lidx_m1*n_state_liou,0,n_state_liou,1);
        auto j_float = static_cast<REAL_TYPE(T)>(j[lidx][lk_u[k]]);
        tmp_Phi.noalias() += std::sqrt(j_float)*S_u.coeff(k)*rho_nm1;
        if (A_u.coeff(k) != zero<T>()) {
          tmp_Psi.noalias() -= std::sqrt(j_float)*A_u.coeff(k)*rho_nm1;
        }
      }
      
      tmp.noalias() += Phi_impl_u*tmp_Phi;
      tmp.noalias() += Psi_impl_u*tmp_Psi;
    }
    drho_dt_n *= beta;
    drho_dt_n.noalias() += -alpha*tmp;
  }
}


// template<typename T,
// template<typename, int> class matrix_type,
//   int NumState>
// void heom_ll<T, matrix_type>::ConstructCommutator(
//     lil_matrix<T>& x,
//     T coeff_l,
//     T coeff_r,
//     std::function<void(int)> callback,
//     int interval_callback) {
// }


// template<typename T,
//   template<typename, int> class matrix_type,
// int NumState>
// void heom_ll<T, matrix_type>::ApplyCommutator(ref<dense_vector<T>> rho) {
// }


//========================================================================
// HEOM Module by using hierarchical-Liouville space expression
//========================================================================


template<typename T,
         template<typename, int> class matrix_type,
         int num_state>
void heom_lh<T, matrix_type, num_state>::init_aux_vars() {
  heom_l<T>::init_aux_vars();
  
  R_heom.set_shape(this->n_hrchy*this->n_state_liou, this->n_hrchy*this->n_state_liou);
  
  for (int lidx = 0; lidx < this->n_hrchy; ++lidx) {
    for (int a = 0; a < this->n_state_liou; ++a) {
      // -1 terms
      for (int u = 0; u < this->n_noise; ++u) {
        for (int k = 0; k < this->len_gamma[u]; ++k) {
          int lidx_m1 = this->hs.ptr_m1[lidx][this->lk[u][k]];
          if (lidx_m1 == this->hs.ptr_void) continue;
          try {
            for (auto& Theta_kv: this->Theta[u][k].data[a]) {
              int b = Theta_kv.first;
              T val = Theta_kv.second;
              val *= std::sqrt(static_cast<REAL_TYPE(T)>(this->hs.j[lidx][this->lk[u][k]]));
              // val *= static_cast<REAL_TYPE(T)>(this->hs.j[lidx][this->lk[u][k]]);
              if (val != zero<T>()) {
                this->R_heom.push(lidx*this->n_state_liou + a,
                                  lidx_m1*this->n_state_liou + b,
                                  val);
              }
            }
          } catch (std::out_of_range&) {}
        }
      }
      
      // 0 terms
      this->R_heom.push(lidx*this->n_state_liou + a,
                        lidx*this->n_state_liou + a,
                        this->jgamma_diag[lidx]);
      
      for (int u = 0; u < this->n_noise; ++u) {
        for (int jj = 0; jj < this->gamma_offdiag[u].rows(); ++jj) {
          for (int ptr = this->gamma_offdiag[u].outerIndexPtr()[jj];
               ptr < this->gamma_offdiag[u].outerIndexPtr()[jj+1]; ++ptr) {
            int k = this->gamma_offdiag[u].innerIndexPtr()[ptr];
            const T& val = this->gamma_offdiag[u].valuePtr()[ptr];
            int lidx_m1j, lidx_m1jp1k;
            if ((lidx_m1j = this->hs.ptr_m1[lidx][this->lk[u][jj]])
                != this->hs.ptr_void
                && (lidx_m1jp1k = this->hs.ptr_p1[lidx_m1j][this->lk[u][k]])
                != this->hs.ptr_void)  {
              auto j_float = static_cast<REAL_TYPE(T)>(this->hs.j[lidx][this->lk[u][jj]]);
              auto k_float = static_cast<REAL_TYPE(T)>(this->hs.j[lidx][this->lk[u][k]]);
              this->R_heom.push(lidx*this->n_state_liou + a,
                                lidx_m1jp1k*this->n_state_liou + a,
                                val*std::sqrt(j_float*(k_float + 1)));
            }
          }
        }
      }
      
      try {
        for (auto& R_kv: this->R_heom_0.data[a]) {
          int b = R_kv.first;
          T val = R_kv.second;
          this->R_heom.push(lidx*this->n_state_liou + a,
                            lidx*this->n_state_liou + b,
                            val);
        }
      } catch (std::out_of_range&) {
      }

      // +1 terms
      for (int u = 0; u < this->n_noise; ++u) {
        for (int k = 0; k < this->len_gamma[u]; ++k) {
          int lidx_p1 = this->hs.ptr_p1[lidx][this->lk[u][k]];
          if (lidx_p1 == this->hs.ptr_void) continue;
          try {
            for (auto& Phi_kv: this->Phi[u].data[a]) {
              int b = Phi_kv.first;
              T val = Phi_kv.second;
              val *= std::sqrt(static_cast<REAL_TYPE(T)>(this->hs.j[lidx][this->lk[u][k]]+1));
              val *= this->sigma[u].coeff(k);
              if (val != zero<T>()) { 
                this->R_heom.push(lidx*this->n_state_liou + a,
                                  lidx_p1*this->n_state_liou + b,
                                  val);
              }
            }
          } catch (std::out_of_range&) {}
        }
      }
    }
  }

  this->R_heom.template dump<Eigen::Dynamic>(this->R_heom_impl);
  // this->R_heom_impl = static_cast<matrix_type<T>>(this->R_heom);

  // std::ofstream os("tmp3.dat");
  // os << "R_heom" << this->R_heom << std::endl;
}


template<typename T,
         template<typename, int> class matrix_type,
         int num_state>
void heom_lh<T, matrix_type, num_state>::calc_diff(
    ref<dense_vector<T,Eigen::Dynamic>> drho_dt,
    const ref<const dense_vector<T,Eigen::Dynamic>>& rho,
    REAL_TYPE(T) alpha,
    REAL_TYPE(T) beta) {
  drho_dt = -alpha*this->R_heom_impl*rho + beta*drho_dt;
}


// template<typename T,
// template<typename, int> class MatrixType,
// int NumState>
// void heom_lh<T, MatrixType>::ConstructCommutator(
//     lil_matrix<T>& x,
//     T coeff_l,
//     T coeff_r,
//     std::function<void(int)> callback,
//     int interval_callback) {
//   x = x;
  
//   this->X.set_shape(this->n_state_liou, this->n_state_liou);
//   kron_identity_right(coeff_l, x, zero<T>(), this->X);
//   kron_identity_left (coeff_r, x, one<T>(), this->X);
//   this->X.optimize();

//   this->X_hrchy.set_shape(this->n_hrchy*this->n_state_liou,
//                          this->n_hrchy*this->n_state_liou);
  
//   for (int lidx = 0; lidx < this->n_hrchy; ++lidx) {
//     if (lidx % interval_callback == 0) {
//       callback(lidx);
//     }
    
//     for (int a = 0; a < this->n_state_liou; ++a) {
//       try {
//         for (auto& X_kv: this->X.data[a]) {
//           int b = X_kv.first;
//           T val = X_kv.second;
//           this->X_hrchy.Push(lidx*this->n_state_liou + a,
//                              lidx*this->n_state_liou + b,
//                              val);
//         }
//       } catch (std::out_of_range&) {
//       }
//     }
//   }
//   this->X_hrchy_impl = static_cast<MatrixType<T>>(this->X_hrchy);
// }


// template<typename T,
// template<typename, int> class MatrixType,
// int NumState>
// void heom_lh<T, MatrixType>::ApplyCommutator(ref<dense_vector<T>>& rho) {
//   gemv(one<T>(), this->X_hrchy_impl, rho, zero<T>(), this->sub_vector.data());
//   copy(this->size_rho, this->sub_vector.data(), rho);
// }


}

// Explicit instantiations
namespace libheom {

template void heom<complex64>::linearize();
template void heom<complex64>::init();
template void heom<complex128>::linearize();
template void heom<complex128>::init();

#define DECLARE_EXPLICIT_INSTANTIATIONS(qme_type, T, matrix_type, num_state) \
  template void qme_type<T, matrix_type, num_state>::init_aux_vars();   \
  template void qme_type<T, matrix_type, num_state>::calc_diff(                       \
      ref<dense_vector<T, Eigen::Dynamic>> drho_dt, \
      const ref<const dense_vector<T, Eigen::Dynamic>>& rho,     \
      REAL_TYPE(T) alpha, REAL_TYPE(T) beta);
// template void qme_type<T, matrix_type>::ConstructCommutator(            \
//     lil_matrix<T>& x,                                                  \
//     T coeff_l,                                                        \
//     T coeff_r,                                                        \
//     std::function<void(int)> callback,                                \
//     int interval_callback);                                           \
// template void qme_type<T, matrix_type>::ApplyCommutator(ref<dense_vector<T>> rho);

DECLARE_EXPLICIT_INSTANTIATIONS(heom_ll, complex64,  dense_matrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(heom_ll, complex64,  csr_matrix,   Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(heom_ll, complex128, dense_matrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(heom_ll, complex128, csr_matrix,   Eigen::Dynamic);

DECLARE_EXPLICIT_INSTANTIATIONS(heom_lh, complex64,  dense_matrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(heom_lh, complex64,  csr_matrix,   Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(heom_lh, complex128, dense_matrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(heom_lh, complex128, csr_matrix,   Eigen::Dynamic);

DECLARE_EXPLICIT_INSTANTIATIONS(heom_ll, complex64,  dense_matrix, 2);
DECLARE_EXPLICIT_INSTANTIATIONS(heom_ll, complex64,  csr_matrix,   2);
DECLARE_EXPLICIT_INSTANTIATIONS(heom_ll, complex128, dense_matrix, 2);
DECLARE_EXPLICIT_INSTANTIATIONS(heom_ll, complex128, csr_matrix,   2);

DECLARE_EXPLICIT_INSTANTIATIONS(heom_lh, complex64,  dense_matrix, 2);
DECLARE_EXPLICIT_INSTANTIATIONS(heom_lh, complex64,  csr_matrix,   2);
DECLARE_EXPLICIT_INSTANTIATIONS(heom_lh, complex128, dense_matrix, 2);
DECLARE_EXPLICIT_INSTANTIATIONS(heom_lh, complex128, csr_matrix,   2);

}
