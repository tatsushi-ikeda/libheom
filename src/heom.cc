/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
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
void Heom<T>::LinearizeDim() {
  this->hs.n_dim
      = std::accumulate(this->len_gamma.get(),
                        this->len_gamma.get() + this->n_noise, 0);
  this->lk.reset(new std::unique_ptr<int[]>[this->n_noise]);
  
  int ctr_lk = 0;
  for (int u = 0; u < this->n_noise; ++u) {
    this->lk[u].reset(new int[this->len_gamma[u]]);
    for (int k = 0; k < this->len_gamma[u]; ++k) {
      this->lk[u][k] = ctr_lk;
      ++ctr_lk;
    }
  }
}


template<typename T>
void Heom<T>::Initialize() {
  this->size_rho = (this->n_hierarchy+1)*this->n_state*this->n_state;
  this->sub_vector.resize(this->size_rho);
  this->sub_vector.fill(Zero<T>());
}


template<typename T>
void Heom<T>::InitAuxVars(std::function<void(int)> callback) {
  Qme<T>::InitAuxVars(callback);

  this->jgamma_diag.resize(this->n_hierarchy);
  for (int lidx = 0; lidx < this->n_hierarchy; ++lidx) {
    this->jgamma_diag[lidx] = Zero<T>();
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
void HeomL<T>::InitAuxVars(std::function<void(int)> callback) {
  Heom<T>::InitAuxVars(callback);
  
  this->n_state_liou = this->n_state*this->n_state;
  
  this->L.SetShape(this->n_state_liou, this->n_state_liou);
  kron_identity_right(+IUnit<T>(), this->H, Zero<T>(), this->L);
  kron_identity_left (-IUnit<T>(), this->H, One<T>(),  this->L);
  this->L.Optimize();
  
  this->Phi.reset(new LilMatrix<T> [this->n_noise]);
  this->Psi.reset(new LilMatrix<T> [this->n_noise]);
  this->Theta.reset(new std::unique_ptr<LilMatrix<T>[]> [this->n_noise]);
  this->Xi.reset(new LilMatrix<T> [this->n_noise]);
  
  for (int u = 0; u < this->n_noise; ++u) {
    this->Phi[u].SetShape(this->n_state_liou, this->n_state_liou);
    kron_identity_right(+IUnit<T>(), this->V[u], Zero<T>(), this->Phi[u]);
    kron_identity_left (-IUnit<T>(), this->V[u], One<T>(),  this->Phi[u]);
    this->Phi[u].Optimize();
    
    this->Psi[u].SetShape(this->n_state_liou, this->n_state_liou);
    kron_identity_right(Frac<T>(1,1), this->V[u], Zero<T>(), this->Psi[u]);
    kron_identity_left (Frac<T>(1,1), this->V[u], One<T>(),  this->Psi[u]);
    this->Psi[u].Optimize();

    this->Theta[u].reset(new LilMatrix<T>[this->len_gamma[u]]);
    for (int k = 0; k < this->len_gamma[u]; ++k) {
      this->Theta[u][k].SetShape(this->n_state_liou, this->n_state_liou);
      axpy(+this->S[u].coeff(k), this->Phi[u], this->Theta[u][k]);
      axpy(-this->A[u].coeff(k), this->Psi[u], this->Theta[u][k]);
      this->Theta[u][k].Optimize();
    }

    this->Xi[u].SetShape(this->n_state_liou, this->n_state_liou);
    gemm(-this->S_delta[u], this->Phi[u], this->Phi[u], Zero<T>(), this->Xi[u]);
    this->Xi[u].Optimize();
  }

  this->R_heom_0.SetShape(this->n_state_liou, this->n_state_liou);
  axpy(One<T>(), this->L, this->R_heom_0);
  for (int u = 0; u < this->n_noise; ++u) {
    axpy(One<T>(), this->Xi[u], this->R_heom_0);
  }
  // this->L.Optimize();

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
         template<typename, int> class MatrixType,
         int NumState>
void HeomLL<T, MatrixType, NumState>::InitAuxVars(
    std::function<void(int)> callback) {
  HeomL<T>::InitAuxVars(callback);

  this->L.template Dump<NumStateLiou>(this->L_impl);

  // this->L_impl= static_cast<MatrixType<T>>(this->L);

  this->Phi_impl.reset(new MatrixLiou [this->n_noise]);
  this->Psi_impl.reset(new MatrixLiou [this->n_noise]);
  this->Xi_impl.reset(new MatrixLiou [this->n_noise]);

  for (int u = 0; u < this->n_noise; ++u) {
    this->Phi[u].template Dump<NumStateLiou>(this->Phi_impl[u]);
    this->Psi[u].template Dump<NumStateLiou>(this->Psi_impl[u]);
    this->Xi[u].template Dump<NumStateLiou>(this->Xi_impl[u]);
  }
  this->R_heom_0.template Dump<NumStateLiou>(this->R_heom_0_impl);
}


template<typename T,
         template<typename, int> class MatrixType,
         int NumState>
void HeomLL<T, MatrixType, NumState>::CalcDiff(
    Ref<DenseVector<T,Eigen::Dynamic>> drho_dt,
    const Ref<const DenseVector<T,Eigen::Dynamic>>& rho,
    REAL_TYPE(T) alpha,
    REAL_TYPE(T) beta) {
  auto n_hierarchy    = this->n_hierarchy;
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

  DenseVector<T,NumStateLiou> tmp(this->n_state_liou);
  DenseVector<T,NumStateLiou> tmp_Phi(this->n_state_liou);
  DenseVector<T,NumStateLiou> tmp_Psi(this->n_state_liou);

  for (int lidx = 0; lidx < n_hierarchy; ++lidx) {
    // auto rho_n     = rho.block(lidx*n_state_liou,0,n_state_liou,1);
    // auto drho_dt_n = drho_dt.block(lidx*n_state_liou,0,n_state_liou,1);
    auto rho_n     = Block<NumStateLiou,1>::value(rho, lidx*n_state_liou,0,n_state_liou,1);
    auto drho_dt_n = Block<NumStateLiou,1>::value(drho_dt, lidx*n_state_liou,0,n_state_liou,1);

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
            auto rho_m1jp1k = Block<NumStateLiou,1>::value(
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
        auto rho_np1 = Block<NumStateLiou,1>::value(rho, lidx_p1*n_state_liou,0,n_state_liou,1);
        auto j_float = static_cast<REAL_TYPE(T)>(j[lidx][lk_u[k]]);
        tmp_Phi.noalias() += sigma_u.coeff(k)*std::sqrt(j_float + 1)*rho_np1;
      }

      // -1 terms
      for (int k = 0; k < len_gamma_u; ++k) {
        int lidx_m1 = ptr_m1[lidx][lk_u[k]];
        auto rho_nm1 = Block<NumStateLiou,1>::value(rho, lidx_m1*n_state_liou,0,n_state_liou,1);
        auto j_float = static_cast<REAL_TYPE(T)>(j[lidx][lk_u[k]]);
        tmp_Phi.noalias() += std::sqrt(j_float)*S_u.coeff(k)*rho_nm1;
        if (A_u.coeff(k) != Zero<T>()) {
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
// template<typename, int> class MatrixType,
//   int NumState>
// void HeomLL<T, MatrixType>::ConstructCommutator(
//     LilMatrix<T>& x,
//     T coeff_l,
//     T coeff_r,
//     std::function<void(int)> callback,
//     int interval_callback) {
// }


// template<typename T,
//   template<typename, int> class MatrixType,
// int NumState>
// void HeomLL<T, MatrixType>::ApplyCommutator(Ref<DenseVector<T>> rho) {
// }


//========================================================================
// HEOM Module by using hierarchical-Liouville space expression
//========================================================================


template<typename T,
         template<typename, int> class MatrixType,
         int NumState>
void HeomLH<T, MatrixType, NumState>::InitAuxVars(std::function<void(int)> callback) {
  HeomL<T>::InitAuxVars(callback);
  
  R_heom.SetShape(this->n_hierarchy*this->n_state_liou, this->n_hierarchy*this->n_state_liou);
  
  for (int lidx = 0; lidx < this->n_hierarchy; ++lidx) {
    // if (lidx % interval_callback == 0) {
    callback(lidx);
    // }

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
              if (val != Zero<T>()) {
                this->R_heom.Push(lidx*this->n_state_liou + a,
                                  lidx_m1*this->n_state_liou + b,
                                  val);
              }
            }
          } catch (std::out_of_range&) {}
        }
      }
      
      // 0 terms
      this->R_heom.Push(lidx*this->n_state_liou + a,
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
              this->R_heom.Push(lidx*this->n_state_liou + a,
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
          this->R_heom.Push(lidx*this->n_state_liou + a,
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
              if (val != Zero<T>()) { 
                this->R_heom.Push(lidx*this->n_state_liou + a,
                                  lidx_p1*this->n_state_liou + b,
                                  val);
              }
            }
          } catch (std::out_of_range&) {}
        }
      }
    }
  }

  this->R_heom.template Dump<Eigen::Dynamic>(this->R_heom_impl);
  // this->R_heom_impl = static_cast<MatrixType<T>>(this->R_heom);

  // std::ofstream os("tmp3.dat");
  // os << "R_heom" << this->R_heom << std::endl;
}


template<typename T,
         template<typename, int> class MatrixType,
         int NumState>
void HeomLH<T, MatrixType, NumState>::CalcDiff(
    Ref<DenseVector<T,Eigen::Dynamic>> drho_dt,
    const Ref<const DenseVector<T,Eigen::Dynamic>>& rho,
    REAL_TYPE(T) alpha,
    REAL_TYPE(T) beta) {
  drho_dt = -alpha*this->R_heom_impl*rho + beta*drho_dt;
}


// template<typename T,
// template<typename, int> class MatrixType,
// int NumState>
// void HeomLH<T, MatrixType>::ConstructCommutator(
//     LilMatrix<T>& x,
//     T coeff_l,
//     T coeff_r,
//     std::function<void(int)> callback,
//     int interval_callback) {
//   x = x;
  
//   this->X.SetShape(this->n_state_liou, this->n_state_liou);
//   kron_identity_right(coeff_l, x, Zero<T>(), this->X);
//   kron_identity_left (coeff_r, x, One<T>(), this->X);
//   this->X.Optimize();

//   this->X_hrchy.SetShape(this->n_hierarchy*this->n_state_liou,
//                          this->n_hierarchy*this->n_state_liou);
  
//   for (int lidx = 0; lidx < this->n_hierarchy; ++lidx) {
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
// void HeomLH<T, MatrixType>::ApplyCommutator(Ref<DenseVector<T>>& rho) {
//   gemv(One<T>(), this->X_hrchy_impl, rho, Zero<T>(), this->sub_vector.data());
//   copy(this->size_rho, this->sub_vector.data(), rho);
// }


}

// Explicit instantiations
namespace libheom {
template void Heom<complex64>::LinearizeDim();
template void Heom<complex64>::Initialize();
template void Heom<complex128>::LinearizeDim();
template void Heom<complex128>::Initialize();

#define DECLARE_EXPLICIT_INSTANTIATIONS(QmeType, T, MatrixType, NumState) \
  template void QmeType<T, MatrixType, NumState>::InitAuxVars(                   \
      std::function<void(int)> callback);                               \
  template void QmeType<T, MatrixType, NumState>::CalcDiff(                       \
      Ref<DenseVector<T, Eigen::Dynamic>> drho_dt, \
      const Ref<const DenseVector<T, Eigen::Dynamic>>& rho,     \
      REAL_TYPE(T) alpha, REAL_TYPE(T) beta);
// template void QmeType<T, MatrixType>::ConstructCommutator(            \
//     LilMatrix<T>& x,                                                  \
//     T coeff_l,                                                        \
//     T coeff_r,                                                        \
//     std::function<void(int)> callback,                                \
//     int interval_callback);                                           \
// template void QmeType<T, MatrixType>::ApplyCommutator(Ref<DenseVector<T>> rho);

DECLARE_EXPLICIT_INSTANTIATIONS(HeomLL, complex64,  DenseMatrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(HeomLL, complex64,  CsrMatrix,   Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(HeomLL, complex128, DenseMatrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(HeomLL, complex128, CsrMatrix,   Eigen::Dynamic);

DECLARE_EXPLICIT_INSTANTIATIONS(HeomLH, complex64,  DenseMatrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(HeomLH, complex64,  CsrMatrix,   Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(HeomLH, complex128, DenseMatrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(HeomLH, complex128, CsrMatrix,   Eigen::Dynamic);

DECLARE_EXPLICIT_INSTANTIATIONS(HeomLL, complex64,  DenseMatrix, 2);
DECLARE_EXPLICIT_INSTANTIATIONS(HeomLL, complex64,  CsrMatrix,   2);
DECLARE_EXPLICIT_INSTANTIATIONS(HeomLL, complex128, DenseMatrix, 2);
DECLARE_EXPLICIT_INSTANTIATIONS(HeomLL, complex128, CsrMatrix,   2);

DECLARE_EXPLICIT_INSTANTIATIONS(HeomLH, complex64,  DenseMatrix, 2);
DECLARE_EXPLICIT_INSTANTIATIONS(HeomLH, complex64,  CsrMatrix,   2);
DECLARE_EXPLICIT_INSTANTIATIONS(HeomLH, complex128, DenseMatrix, 2);
DECLARE_EXPLICIT_INSTANTIATIONS(HeomLH, complex128, CsrMatrix,   2);

}
