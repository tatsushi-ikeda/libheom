/*
 * LibHEOM: Copyright (c) Tatsushi Ikeda
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
    redfield<T>& rf,
    int u,
    real_t<T> omega) {

  if (rf.use_corr_func && rf.use_corr_func[u]) {
    return rf.corr_func[u](omega);
  } else {
    
    Eigen::SparseMatrix<T, Eigen::RowMajor> I(rf.len_gamma[u],rf.len_gamma[u]);
    I.setIdentity();
    Eigen::SparseLU<Eigen::SparseMatrix<T, Eigen::RowMajor>> solver;
    solver.compute(rf.gamma[u] - i_unit<T>()*omega*I);
    if (solver.info() != Eigen::Success) {
      std::cerr << "[Error] LU decomposition failed. " << std::endl;
      std::exit(1);
      return zero<T>();
    }
    return static_cast<T>(rf.sigma[u].transpose()*(rf.s[u] + i_unit<T>()*rf.a[u])*solver.solve(rf.phi_0[u])) + rf.S_delta[u];
    
  }
}


template<typename T>
void redfield<T>::init_aux_vars() {
  qme<T>::init_aux_vars();
  
  this->Lambda.reset(new lil_matrix<T>[this->n_noise]);
  for (int s = 0; s < this->n_noise; ++s) {
    this->Lambda[s].set_shape(this->n_state, this->n_state);
    for (auto& V_ijv : this->V[s].data) {
      int i = V_ijv.first;
      for (auto& V_jv: V_ijv.second) {
        int j = V_jv.first;
        T val = V_jv.second;
        if (val != zero<T>()) {
          real_t<T> omega_ji;
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
// redfield Module by using Hilbert space expression
//========================================================================


template<typename T,
         template <typename, int> class matrix_type,
         int num_state>
void redfield_h<T, matrix_type, num_state>::init_aux_vars() {
  redfield<T>::init_aux_vars();

  this->H.template dump<num_state>(this->H_impl);

  this->V_impl.reset(new matrix_hilb[this->n_noise]);
  this->Lambda_impl.reset(new matrix_hilb[this->n_noise]);
  this->Lambda_dagger_impl.reset(new matrix_hilb[this->n_noise]);
  
  for (int s = 0; s < this->n_noise; ++s) {
    this->V[s].template dump<num_state>(this->V_impl[s]);
    this->Lambda[s].template dump<num_state>(this->Lambda_impl[s]);
    auto tmp = this->Lambda[s].hermite_conjugate();
    tmp.template dump<num_state>(this->Lambda_dagger_impl[s]);
  }
}


// template<typename T, template <typename> class matrix_type>
// void redfield_h<T, matrix_type>::ConstructCommutator(
//     lil_matrix<T>& x,
//     T coef_l,
//     T coef_r,
//     std::function<void(int)> callback,
//     int interval_callback) {
//   this->X_impl = static_cast<matrix_type<T>>(this->x);
//   this->coef_l_X = coef_l;
//   this->coef_r_X = coef_r;
// }


template<typename T,
         template <typename, int> class matrix_type,
         int num_state>
void redfield_h<T, matrix_type, num_state>::calc_diff(
    ref<dense_vector<T,Eigen::Dynamic>> drho_dt_raw,
    const ref<const dense_vector<T,Eigen::Dynamic>>& rho_raw,
    real_t<T> alpha,
    real_t<T> beta) {
  auto n_state = this->n_state;
  
  auto rho     = Eigen::Map<const dense_matrix<T,num_state>>(rho_raw.data(),n_state,n_state);
  auto drho_dt = Eigen::Map<dense_matrix<T,num_state>>(drho_dt_raw.data(),n_state,n_state);
  dense_matrix<T,num_state> tmp(n_state,n_state);

  drho_dt *= beta;
  drho_dt.noalias() += -alpha*i_unit<T>()*this->H_impl*rho;
  drho_dt.noalias() += +alpha*i_unit<T>()*rho*this->H_impl;
  
  for (int s = 0; s < this->n_noise; ++s) {
    tmp.noalias()  = +i_unit<T>()*this->Lambda_impl[s]*rho;
    tmp.noalias() += -i_unit<T>()*rho*this->Lambda_dagger_impl[s];
    drho_dt.noalias() += alpha*i_unit<T>()*this->V_impl[s]*tmp;
    drho_dt.noalias() -= alpha*i_unit<T>()*tmp*this->V_impl[s];
  }
}


// template<typename T, template <typename, int> class matrix_type, int num_state>
// void redfield_h<T, matrix_type, num_state>::ApplyCommutator(T* rho_raw) {
//   DenseMatrixWrapper<T> rho(this->n_state, this->n_state, rho_raw);
//   DenseMatrixWrapper<T> sub(this->n_state, this->n_state, this->sub_vector.data());
  
//   gemm(this->coef_l_X, this->X_impl, rho, zero<T>(), sub);
//   gemm(this->coef_r_X, rho, this->X_impl, one <T>(), sub);
// }


//========================================================================
// redfield Module by using Liouville space expression
//========================================================================


template<typename T,
         template <typename, int> class matrix_type,
         int num_state>
void redfield_l<T, matrix_type, num_state>::init_aux_vars() {
  redfield<T>::init_aux_vars();
  
  this->n_state_liou = this->n_state*this->n_state;
  
  this->L.set_shape(this->n_state_liou, this->n_state_liou);
  kron_identity_right(+i_unit<T>(), this->H, zero<T>(), this->L);
  kron_identity_left (-i_unit<T>(), this->H, one<T>(), this->L);
  
  this->Phi.  reset(new lil_matrix<T>[this->n_noise]);
  this->Theta.reset(new lil_matrix<T>[this->n_noise]);
  
  for (int s = 0; s < this->n_noise; ++s){
    this->Phi[s].set_shape(this->n_state_liou, this->n_state_liou);
    kron_identity_right(+i_unit<T>(), this->V[s], zero<T>(), this->Phi[s]);
    kron_identity_left (-i_unit<T>(), this->V[s], one<T>(),  this->Phi[s]);

    this->Theta[s].set_shape(this->n_state_liou, this->n_state_liou);
    kron_identity_right(+i_unit<T>(), this->Lambda[s],
                        zero<T>(), this->Theta[s]);
    kron_identity_left (-i_unit<T>(), this->Lambda[s].hermite_conjugate(),
                        one<T>(),  this->Theta[s]);
  }

  this->R_redfield.set_shape(this->n_state_liou, this->n_state_liou);
  axpy(one<T>(), this->L, this->R_redfield);
  for (int s = 0; s < this->n_noise; ++s) {
    gemm(-one<T>(), this->Phi[s], this->Theta[s], one<T>(), this->R_redfield);
  }
  this->R_redfield.optimize();

  R_redfield.template dump<num_state_liou>(this->R_redfield_impl);
}


// template<typename T, template <typename> class matrix_type>
// void redfield_l<T, matrix_type>::ConstructCommutator(
//     lil_matrix<T>& x,
//     T coef_l,
//     T coef_r,
//     std::function<void(int)> callback,
//     int interval_callback) {
//   this->X_impl = static_cast<matrix_type<T>>(this->x);
//   this->coef_l_X = coef_l;
//   this->coef_r_X = coef_r;
// }


template<typename T,
         template <typename, int> class matrix_type,
         int num_state>
void redfield_l<T, matrix_type, num_state>::calc_diff(
    ref<dense_vector<T,Eigen::Dynamic>> drho_dt_raw,
    const ref<const dense_vector<T,Eigen::Dynamic>>& rho_raw,
    real_t<T> alpha,
    real_t<T> beta) {
  auto n_state_liou   = this->n_state_liou;
  auto rho     = block<num_state_liou,1>::value(rho_raw, 0,0,n_state_liou,1);
  auto drho_dt = block<num_state_liou,1>::value(drho_dt_raw,0,0,n_state_liou,1);

  drho_dt *= beta;
  drho_dt.noalias() += -alpha*this->R_redfield_impl*rho;
}


// template<typename T, template <typename, int> class matrix_type, int num_state>
// void redfield_l<T, matrix_type, num_state>::ApplyCommutator(T* rho_raw) {
//   DenseMatrixWrapper<T> rho(this->n_state, this->n_state, rho_raw);
//   DenseMatrixWrapper<T> sub(this->n_state, this->n_state, this->sub_vector.data());
  
//   gemm(this->coef_l_X, this->X_impl, rho, zero<T>(), sub);
//   gemm(this->coef_r_X, rho, this->X_impl, one <T>(), sub);
// }


}

// Explicit instantiations
namespace libheom {

#define DECLARE_EXPLICIT_INSTANTIATIONS(qme_type, T, matrix_type, num_state) \
  template void qme_type<T, matrix_type, num_state>::init_aux_vars();   \
  template void qme_type<T, matrix_type, num_state>::calc_diff(                       \
      ref<dense_vector<T, Eigen::Dynamic>> drho_dt, \
      const ref<const dense_vector<T, Eigen::Dynamic>>& rho,     \
      real_t<T> alpha, real_t<T> beta);
// template void qme_type<T, matrix_type>::ConstructCommutator(            \
//     lil_matrix<T>& x,                                                  \
//     T coef_l,                                                         \
//     T coef_r,                                                         \
//     std::function<void(int)> callback,                                \
//     int interval_callback);                                           \
// template void qme_type<T, matrix_type>::ApplyCommutator(ref<dense_vector<T>> rho);

DECLARE_EXPLICIT_INSTANTIATIONS(redfield_h, complex64,  dense_matrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(redfield_h, complex64,  csr_matrix,   Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(redfield_h, complex128, dense_matrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(redfield_h, complex128, csr_matrix,   Eigen::Dynamic);

DECLARE_EXPLICIT_INSTANTIATIONS(redfield_l, complex64,  dense_matrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(redfield_l, complex64,  csr_matrix,   Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(redfield_l, complex128, dense_matrix, Eigen::Dynamic);
DECLARE_EXPLICIT_INSTANTIATIONS(redfield_l, complex128, csr_matrix,   Eigen::Dynamic);

DECLARE_EXPLICIT_INSTANTIATIONS(redfield_h, complex64,  dense_matrix, 2);
DECLARE_EXPLICIT_INSTANTIATIONS(redfield_h, complex64,  csr_matrix,   2);
DECLARE_EXPLICIT_INSTANTIATIONS(redfield_h, complex128, dense_matrix, 2);
DECLARE_EXPLICIT_INSTANTIATIONS(redfield_h, complex128, csr_matrix,   2);

DECLARE_EXPLICIT_INSTANTIATIONS(redfield_l, complex64,  dense_matrix, 2);
DECLARE_EXPLICIT_INSTANTIATIONS(redfield_l, complex64,  csr_matrix,   2);
DECLARE_EXPLICIT_INSTANTIATIONS(redfield_l, complex128, dense_matrix, 2);
DECLARE_EXPLICIT_INSTANTIATIONS(redfield_l, complex128, csr_matrix,   2);

}
