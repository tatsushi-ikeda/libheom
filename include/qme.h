/*
 * LibHEOM, Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef QME_H
#define QME_H

#include <memory>
#include <functional>

#include <Eigen/Core>

#include "type.h"
#include "lil_matrix.h"
#include "dense_matrix.h"

namespace libheom
{


constexpr inline int n_state_prod(int a, int b)
{
  return ((a == Eigen::Dynamic) || (b == Eigen::Dynamic)) ? Eigen::Dynamic : a*b;
}


template <typename T,
          int P, int Q,
          bool static_flag        = ((P != Eigen::Dynamic) && (Q != Eigen::Dynamic)),
          bool static_vector_flag = ((P != Eigen::Dynamic) && (Q == 1))>
class blk
{
 public:
  template<typename matrix_type>
  static inline Eigen::Block<matrix_type,P,Q> value
  /**/(matrix_type& matrix, int i, int j, int p, int q)
  {
    return matrix.template block<P,Q>(i,j);
  }
};


template <typename T, int P, int Q>
class blk<T, P, Q, true, true>
{
 public:
  static inline Eigen::Map<dense_vector<T,P>,alignment<T>()> value
  /**/(ref<dense_vector<T,Eigen::Dynamic>> matrix, int i, int j, int p, int q)
  {
    return dense_vector<T,P>::MapAligned(matrix.data() + i);
  }

  static inline const Eigen::Map<const dense_vector<T,P>,alignment<T>()> value
  /**/(const ref<const dense_vector<T,Eigen::Dynamic>>& matrix, int i, int j, int p, int q)
  {
    return dense_vector<T,P>::MapAligned(matrix.data() + i);
  }
};


template <typename T, int P, int Q>
class blk<T, P, Q, false, false>
{
 public:
  template<typename matrix_type>
  static inline Eigen::Block<matrix_type> value
  /**/(matrix_type& matrix, int i, int j, int p, int q)
  {
    return matrix.block(i,j,p,q);
  }
};


template<typename T>
class qme
{
public:
  int n_state;
  lil_matrix<T> H;
  
  int n_noise;
  std::unique_ptr<lil_matrix<T>[]> V;
  
  std::vector<std::vector<int>> lk;

  std::unique_ptr<int[]> len_gamma;
  std::unique_ptr<Eigen::SparseMatrix<T, Eigen::RowMajor>[]> gamma;
  std::unique_ptr<Eigen::Matrix<T,Eigen::Dynamic,1>[]> phi_0;
  std::unique_ptr<Eigen::Matrix<T,Eigen::Dynamic,1>[]> sigma;
  std::unique_ptr<Eigen::SparseMatrix<T, Eigen::RowMajor>[]> s;
  std::unique_ptr<T[]> S_delta;
  std::unique_ptr<Eigen::SparseMatrix<T, Eigen::RowMajor>[]> a;

  T coef_l_X;
  T coef_r_X;
  lil_matrix<T> x;

  int size_rho;

  dense_vector<T,Eigen::Dynamic> sub_vector;
  
  void alloc_noise
  /**/(int n_noise);
  
  void init
  /**/();
  
  void fin
  /**/();

  void solve
  /**/(ref<dense_vector<T,Eigen::Dynamic>> rho,
       real_t<T> dt__unit,
       real_t<T> dt,
       int interval,
       int count,
       std::function<void(real_t<T>)> callback);

  virtual void calc_diff
  /**/(ref<dense_vector<T,Eigen::Dynamic>> drho_dt,
       const ref<const dense_vector<T,Eigen::Dynamic>>& rho,
       real_t<T> alpha,
       real_t<T> beta) = 0;
  
  virtual void evolve
  /**/(ref<dense_vector<T,Eigen::Dynamic>> rho,
       real_t<T> dt,
       const int steps);
  
  virtual void evolve_1
  /**/(ref<dense_vector<T,Eigen::Dynamic>> rho,
       real_t<T> dt);
  
  // virtual void ConstructCommutator(lil_matrix<T>& x,
  //                                  T coef_l,
  //                                  T coef_r,
  //                                  std::function<void(int)> callback
  //                                  = [](int) { return; },
  //                                  int interval_callback = 1024) = 0;
  
  // virtual void ApplyCommutator(Eigen::ref<dense_vector<T>> rho) = 0;

  void init_aux_vars
  /**/() {};

  qme()                      = default;
  qme(const qme&)            = delete;
  qme& operator=(const qme&) = delete;
  virtual ~qme()             = default;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}

#endif
