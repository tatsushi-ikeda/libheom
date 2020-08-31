/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

// Note: In order to avoid compile errors caused by the two phase name
//       lookup rule regarding template class, all class variables in
//       this source have this-> modifier.

#include "redfield.h"

#include <Eigen/SparseLU>

#include "const.h"
// #include "mkl_wrapper.h"

namespace libheom {

// calculate Fourier-Laplace transform of quantum correlation function.
// The exponent is +1j*omega*t.
template<typename T>
inline T correlation(
    Redfield<T>& rf,
    int u,
    REAL_TYPE(T) omega) {

  if (rf.use_corr_func && rf.use_corr_func[u]) {
    return rf.corr_func[u](omega);
  } else {
    
    Eigen::SparseMatrix<T, Eigen::RowMajor> I(rf.len_gamma[u],rf.len_gamma[u]);
    I.setIdentity();
    Eigen::SparseLU<Eigen::SparseMatrix<T, Eigen::RowMajor>> solver;
    solver.compute(rf.gamma[u] - IUnit<T>()*omega*I);
    if (solver.info() != Eigen::Success) {
      std::cerr << "[Error] LU decomposition failed. " << std::endl;
      std::exit(1);
      return Zero<T>();
    }
    return static_cast<T>(rf.sigma[u].transpose()*(rf.s[u] + IUnit<T>()*rf.a[u])*solver.solve(rf.phi_0[u])) + rf.S_delta[u];
    
  }
}


template<typename T>
void Redfield<T>::InitAuxVars(std::function<void(int)> callback) {
  Qme<T>::InitAuxVars(callback);
  
  this->Lambda.reset(new LilMatrix<T>[this->n_noise]);
  for (int s = 0; s < this->n_noise; ++s) {
    this->Lambda[s].SetShape(this->n_state, this->n_state);
    for (auto& V_ijv : this->V[s].data) {
      int i = V_ijv.first;
      for (auto& V_jv: V_ijv.second) {
        int j = V_jv.first;
        T val = V_jv.second;
        if (val != Zero<T>()) {
          REAL_TYPE(T) omega_ji;
          try {
            omega_ji = std::real(this->H.data[j][j] - this->H.data[i][i]);
          } catch (std::out_of_range&) {
            continue;
          }
          T corr = correlation(*this, s, omega_ji);
          this->Lambda[s].data[i][j] += val*corr;
        }
      }
    }
  }
}


//========================================================================
// Redfield Module by using Hilbert space expression
//========================================================================


template<typename T,
         template <typename, int> class MatrixType,
         int NumState>
void RedfieldH<T, MatrixType, NumState>::InitAuxVars(
    std::function<void(int)> callback) {
  Redfield<T>::InitAuxVars(callback);

  this->H.template Dump<NumState>(this->H_impl);

  this->V_impl.reset(new MatrixHilb[this->n_noise]);
  this->Lambda_impl.reset(new MatrixHilb[this->n_noise]);
  this->Lambda_dagger_impl.reset(new MatrixHilb[this->n_noise]);
  
  for (int s = 0; s < this->n_noise; ++s) {
    this->V[s].template Dump<NumState>(this->V_impl[s]);
    this->Lambda[s].template Dump<NumState>(this->Lambda_impl[s]);
    auto tmp = this->Lambda[s].HermiteConjugate();
    tmp.template Dump<NumState>(this->Lambda_dagger_impl[s]);
  }
}


// template<typename T, template <typename> class MatrixType>
// void RedfieldH<T, MatrixType>::ConstructCommutator(
//     LilMatrix<T>& x,
//     T coef_l,
//     T coef_r,
//     std::function<void(int)> callback,
//     int interval_callback) {
//   this->X_impl = static_cast<MatrixType<T>>(this->x);
//   this->coef_l_X = coef_l;
//   this->coef_r_X = coef_r;
// }


template<typename T,
         template <typename, int> class MatrixType,
         int NumState>
void RedfieldH<T, MatrixType, NumState>::CalcDiff(
    Ref<DenseVector<T,Eigen::Dynamic>> drho_dt_raw,
    const Ref<const DenseVector<T,Eigen::Dynamic>>& rho_raw,
    REAL_TYPE(T) alpha,
    REAL_TYPE(T) beta) {
  auto n_state = this->n_state;
  
  auto rho     = Eigen::Map<const DenseMatrix<T,NumState>>(rho_raw.data(),n_state,n_state);
  auto drho_dt = Eigen::Map<DenseMatrix<T,NumState>>(drho_dt_raw.data(),n_state,n_state);
  DenseMatrix<T,NumState> tmp(n_state,n_state);

  drho_dt *= beta;
  drho_dt.noalias() += -alpha*IUnit<T>()*this->H_impl*rho;
  drho_dt.noalias() += +alpha*IUnit<T>()*rho*this->H_impl;
  
  for (int s = 0; s < this->n_noise; ++s) {
    tmp.noalias()  = +IUnit<T>()*this->Lambda_impl[s]*rho;
    tmp.noalias() += -IUnit<T>()*rho*this->Lambda_dagger_impl[s];
    drho_dt.noalias() += alpha*IUnit<T>()*this->V_impl[s]*tmp;
    drho_dt.noalias() -= alpha*IUnit<T>()*tmp*this->V_impl[s];
  }
}


// template<typename T, template <typename, int> class MatrixType, int NumState>
// void RedfieldH<T, MatrixType, NumState>::ApplyCommutator(T* rho_raw) {
//   DenseMatrixWrapper<T> rho(this->n_state, this->n_state, rho_raw);
//   DenseMatrixWrapper<T> sub(this->n_state, this->n_state, this->sub_vector.data());
  
//   gemm(this->coef_l_X, this->X_impl, rho, Zero<T>(), sub);
//   gemm(this->coef_r_X, rho, this->X_impl, One <T>(), sub);
// }


//========================================================================
// Redfield Module by using Liouville space expression
//========================================================================


template<typename T,
         template <typename, int> class MatrixType,
         int NumState>
void RedfieldL<T, MatrixType, NumState>::InitAuxVars(
    std::function<void(int)> callback) {
  Redfield<T>::InitAuxVars(callback);
  
  this->n_state_liou = this->n_state*this->n_state;
  
  this->L.SetShape(this->n_state_liou, this->n_state_liou);
  kron_identity_right(+IUnit<T>(), this->H, Zero<T>(), this->L);
  kron_identity_left (-IUnit<T>(), this->H, One<T>(), this->L);
  
  this->Phi.  reset(new LilMatrix<T>[this->n_noise]);
  this->Theta.reset(new LilMatrix<T>[this->n_noise]);
  
  for (int s = 0; s < this->n_noise; ++s){
    this->Phi[s].SetShape(this->n_state_liou, this->n_state_liou);
    kron_identity_right(+IUnit<T>(), this->V[s], Zero<T>(), this->Phi[s]);
    kron_identity_left (-IUnit<T>(), this->V[s], One<T>(),  this->Phi[s]);

    this->Theta[s].SetShape(this->n_state_liou, this->n_state_liou);
    kron_identity_right(+IUnit<T>(), this->Lambda[s],
                        Zero<T>(), this->Theta[s]);
    kron_identity_left (-IUnit<T>(), this->Lambda[s].HermiteConjugate(),
                        One<T>(),  this->Theta[s]);
  }

  this->R_redfield.SetShape(this->n_state_liou, this->n_state_liou);
  axpy(One<T>(), this->L, this->R_redfield);
  for (int s = 0; s < this->n_noise; ++s) {
    gemm(-One<T>(), this->Phi[s], this->Theta[s], One<T>(), this->R_redfield);
  }
  this->R_redfield.Optimize();

  R_redfield.template Dump<NumStateLiou>(this->R_redfield_impl);
}


// template<typename T, template <typename> class MatrixType>
// void RedfieldL<T, MatrixType>::ConstructCommutator(
//     LilMatrix<T>& x,
//     T coef_l,
//     T coef_r,
//     std::function<void(int)> callback,
//     int interval_callback) {
//   this->X_impl = static_cast<MatrixType<T>>(this->x);
//   this->coef_l_X = coef_l;
//   this->coef_r_X = coef_r;
// }


template<typename T,
         template <typename, int> class MatrixType,
         int NumState>
void RedfieldL<T, MatrixType, NumState>::CalcDiff(
    Ref<DenseVector<T,Eigen::Dynamic>> drho_dt_raw,
    const Ref<const DenseVector<T,Eigen::Dynamic>>& rho_raw,
    REAL_TYPE(T) alpha,
    REAL_TYPE(T) beta) {
  auto n_state_liou   = this->n_state_liou;
  auto rho     = Block<NumStateLiou,1>::value(rho_raw, 0,0,n_state_liou,1);
  auto drho_dt = Block<NumStateLiou,1>::value(drho_dt_raw,0,0,n_state_liou,1);

  drho_dt *= beta;
  drho_dt.noalias() += -alpha*this->R_redfield_impl*rho;
}


// template<typename T, template <typename, int> class MatrixType, int NumState>
// void RedfieldL<T, MatrixType, NumState>::ApplyCommutator(T* rho_raw) {
//   DenseMatrixWrapper<T> rho(this->n_state, this->n_state, rho_raw);
//   DenseMatrixWrapper<T> sub(this->n_state, this->n_state, this->sub_vector.data());
  
//   gemm(this->coef_l_X, this->X_impl, rho, Zero<T>(), sub);
//   gemm(this->coef_r_X, rho, this->X_impl, One <T>(), sub);
// }


}

// Explicit instantiations
namespace libheom {

#define DECLARE_EXPLICIT_INSTANTIATIONS(QmeType, T, MatrixType, NumState) \
  template void QmeType<T, MatrixType, NumState>::InitAuxVars(                   \
      std::function<void(int)> callback);                               \
  template void QmeType<T, MatrixType, NumState>::CalcDiff(                       \
      Ref<DenseVector<T, Eigen::Dynamic>> drho_dt, \
      const Ref<const DenseVector<T, Eigen::Dynamic>>& rho,     \
      REAL_TYPE(T) alpha, REAL_TYPE(T) beta);
// template void QmeType<T, MatrixType>::ConstructCommutator(            \
//     LilMatrix<T>& x,                                                  \
//     T coef_l,                                                         \
//     T coef_r,                                                         \
//     std::function<void(int)> callback,                                \
//     int interval_callback);                                           \
// template void QmeType<T, MatrixType>::ApplyCommutator(Ref<DenseVector<T>> rho);

DECLARE_EXPLICIT_INSTANTIATIONS(RedfieldH, complex64,  DenseMatrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(RedfieldH, complex64,  CsrMatrix,   Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(RedfieldH, complex128, DenseMatrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(RedfieldH, complex128, CsrMatrix,   Eigen::Dynamic);

DECLARE_EXPLICIT_INSTANTIATIONS(RedfieldL, complex64,  DenseMatrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(RedfieldL, complex64,  CsrMatrix,   Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(RedfieldL, complex128, DenseMatrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(RedfieldL, complex128, CsrMatrix,   Eigen::Dynamic);

DECLARE_EXPLICIT_INSTANTIATIONS(RedfieldH, complex64,  DenseMatrix, 2);
DECLARE_EXPLICIT_INSTANTIATIONS(RedfieldH, complex64,  CsrMatrix,   2);
DECLARE_EXPLICIT_INSTANTIATIONS(RedfieldH, complex128, DenseMatrix, 2);
DECLARE_EXPLICIT_INSTANTIATIONS(RedfieldH, complex128, CsrMatrix,   2);

DECLARE_EXPLICIT_INSTANTIATIONS(RedfieldL, complex64,  DenseMatrix, 2);
DECLARE_EXPLICIT_INSTANTIATIONS(RedfieldL, complex64,  CsrMatrix,   2);
DECLARE_EXPLICIT_INSTANTIATIONS(RedfieldL, complex128, DenseMatrix, 2);
DECLARE_EXPLICIT_INSTANTIATIONS(RedfieldL, complex128, CsrMatrix,   2);

}
