/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
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

namespace libheom {

constexpr inline int n_state_prod(int a, int b) {
  return ((a == Eigen::Dynamic) || (b == Eigen::Dynamic)) ? Eigen::Dynamic : a*b;
}


template <int P, int Q,
          bool static_flag = ((P != Eigen::Dynamic) && (Q != Eigen::Dynamic))>
class Block {
 public:
  template<typename MatrixType>
  static inline Eigen::Block<MatrixType,P,Q> value(MatrixType& matrix, int i, int j, int p, int q) {
    return matrix.template block<P,Q>(i,j);
  }
};

template <int P, int Q>
class Block<P, Q, false> {
 public:
  template<typename MatrixType>
  static inline Eigen::Block<MatrixType> value(MatrixType& matrix, int i, int j, int p, int q) {
    return matrix.block(i,j,p,q);
  }
};


template<typename T>
class Qme {
public:
  int n_state;
  LilMatrix<T> H;
  
  int n_noise;
  std::unique_ptr<LilMatrix<T>[]> V;
  
  std::unique_ptr<std::unique_ptr<int[]>[]> lk;

  std::unique_ptr<int[]> len_gamma;
  std::unique_ptr<Eigen::SparseMatrix<T, Eigen::RowMajor>[]> gamma;
  std::unique_ptr<Eigen::Matrix<T,Eigen::Dynamic,1>[]> phi_0;
  std::unique_ptr<Eigen::Matrix<T,Eigen::Dynamic,1>[]> sigma;
  std::unique_ptr<Eigen::SparseMatrix<T, Eigen::RowMajor>[]> s;
  std::unique_ptr<T[]> S_delta;
  std::unique_ptr<Eigen::SparseMatrix<T, Eigen::RowMajor>[]> a;

  T coef_l_X;
  T coef_r_X;
  LilMatrix<T> x;

  int size_rho;

  DenseVector<T,Eigen::Dynamic> sub_vector;
  
  void AllocateNoise(int n_noise);
  void Initialize();
  void Finalize();

  void TimeEvolution(Ref<DenseVector<T,Eigen::Dynamic>> rho,
                     REAL_TYPE(T) dt__unit,
                     REAL_TYPE(T) dt,
                     int interval,
                     int count,
                     std::function<void(REAL_TYPE(T))> callback);

  virtual void CalcDiff(Ref<DenseVector<T,Eigen::Dynamic>> drho_dt,
                        const Ref<const DenseVector<T,Eigen::Dynamic>>& rho,
                        REAL_TYPE(T) alpha,
                        REAL_TYPE(T) beta) = 0;

  virtual void Evolve(Ref<DenseVector<T,Eigen::Dynamic>> rho,
                      REAL_TYPE(T) dt,
                      const int steps);

  virtual void Evolve1(Ref<DenseVector<T,Eigen::Dynamic>> rho,
                       REAL_TYPE(T) dt);
  
  // virtual void ConstructCommutator(LilMatrix<T>& x,
  //                                  T coef_l,
  //                                  T coef_r,
  //                                  std::function<void(int)> callback
  //                                  = [](int) { return; },
  //                                  int interval_callback = 1024) = 0;
  
  // virtual void ApplyCommutator(Eigen::Ref<DenseVector<T>> rho) = 0;

  void InitAuxVars(std::function<void(int)> callback) {};

  Qme()                      = default;
  Qme(const Qme&)            = delete;
  Qme& operator=(const Qme&) = delete;
  virtual ~Qme()             = default;
};

}

#endif
