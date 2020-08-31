/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef LIL_MATRIX_H
#define LIL_MATRIX_H

#include <limits>
#include <tuple>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "type.h"
#include "blas_wrapper.h"
#include "dense_matrix.h"
#include "csr_matrix.h"
#include "printer.h"

namespace libheom{

// Sparse matrix class by using list in list (LIL) format.
// While the name includes `list', this is implemented by using std::map.
template <typename T>
class LilMatrix {
 public:
  typedef T dtype;
  typedef std::map<int, std::map<int, dtype>> liltype;
  std::tuple<int, int> shape;
  liltype data;

  LilMatrix() {}
  
  LilMatrix(int shape_0, int shape_1)
    : shape(shape_0, shape_1) {}

  void SetShape(int shape_0, int shape_1) {
    std::get<0>(shape) = shape_0;
    std::get<1>(shape) = shape_1;
    data.clear();
  }

  // LilMatrix(const sparse_matrix<T>& csr)
  //   : shape(csr.shape)
  // {
  //   // for (int i = 0; i < csr.indptr.)
  // }

  void Clear() {
    data.clear();
  }

  void Optimize(
      typename T::value_type tol
      = std::numeric_limits<typename T::value_type>::epsilon()) {
    typename T::value_type max = 0;
    for (auto& ijv : data) {
      int i = ijv.first;
      for (auto& jv: ijv.second) {
        int j = jv.first;
        max = std::max(std::abs(jv.second), max);
      }
    }
    if (max == 0) {
      max = 1;
    }

    std::vector<std::tuple<int,int>> dead_idx;
    for (auto& ijv : data) {
      int i = ijv.first;
      for (auto& jv: ijv.second) {
        int j = jv.first;
        if (std::abs(jv.second) <= tol*max) {
          dead_idx.push_back(std::make_tuple(i,j));
        }
      }
    }
 
    for (auto& ij: dead_idx) {
      int i = std::get<0>(ij);
      int j = std::get<1>(ij);
      data.at(i).erase(j);
    }

    std::vector<int> dead_row;
    for (auto& ijv : data) {
      int i = ijv.first;
      if (ijv.second.empty()) {
        dead_row.push_back(i);
      }
    }

    for (auto& i: dead_row) {
      data.erase(i);
    }
  }
  
  void Push(const int row, const int col, const T& value) {
    data[row][col] += value;
  }

  LilMatrix<T> Transpose() const {
    LilMatrix<T> out(std::get<1>(shape), std::get<0>(shape));
    for (auto& data_ijv : data) {
      int i = data_ijv.first;
      for (auto& data_jv: data_ijv.second) {
        int j = data_jv.first;
        out.Push(j,i,data_jv.second);
      }
    }
    return std::move(out);
  }

  LilMatrix<T> HermiteConjugate() const {
    LilMatrix<T> out(std::get<1>(shape), std::get<0>(shape));
    for (auto& data_ijv : data) {
      int i = data_ijv.first;
      for (auto& data_jv: data_ijv.second) {
        int j = data_jv.first;
        out.Push(j,i,std::conj(data_jv.second));
      }
    }
    return std::move(out);
  }


  template<int N = Eigen::Dynamic>
  void Dump(DenseMatrix<T,N>& out) const {
    out.resize(std::get<0>(shape), std::get<1>(shape));
    out.setZero();
    for (auto& data_ijv : data) {
      int i = data_ijv.first;
      for (auto& data_jv: data_ijv.second) {
        int j = data_jv.first;
        out(i,j) = data_jv.second;
      }
    }
  }


  template<int N = Eigen::Dynamic>
  void Dump(CsrMatrix<T,N>& out) const {
    out.resize(std::get<0>(shape), std::get<1>(shape));
    std::vector<Eigen::Triplet<T>> list;
    for (auto& data_ijv : data) {
      int i = data_ijv.first;
      for (auto& data_jv: data_ijv.second) {
        int j = data_jv.first;
        list.push_back(Eigen::Triplet<T>(i,j,data_jv.second));
      }
    }
    out.setFromTriplets(list.begin(), list.end());
    out.makeCompressed();
  }
  
  // T& operator () (int i, int j)
  // {
  //   return &data[i][j];
  // }
  
  // const T& operator () (int i, int j) const
  // {
  //   return &data[i][j];
  // }
};


template <typename T>
std::ostream& operator << (std::ostream& out, const LilMatrix<T>& m) {
  out << "LIL Matrix" << std::endl;
  out << "  internal data:" << std::endl;
  out << "  - shape: "   << ShapePrinter(m.shape) << std::endl;
  out << "  - data: " << std::endl;
  for (auto& m_ijv : m.data) {
    int i = m_ijv.first;
    for (auto& m_jv: m_ijv.second) {
      int j = m_jv.first;
      const T& val = m_jv.second;
      out << "    (" << i << ", " << j << ") = " << val << std::endl;
    }
  }
  return out;
};


template<typename T>
inline void kron_identity_right(const T& alpha,
                                const LilMatrix<T>& x,
                                const T& beta,
                                LilMatrix<T>& y,
                                bool conj=false) {
  if (beta == static_cast<T>(0)) {
    y.data.clear();
  } else {
    for (auto& y_ijv : y.data) {
      for (auto& y_jv: y_ijv.second) {
        y_jv.second *= beta;
      }
    }
  }
  
  int n = std::get<0>(x.shape);
  for (auto& x_ikv : x.data) {
    int i = x_ikv.first;
    for (auto& x_kv: x_ikv.second) {
      int k = x_kv.first;
      const T& val = x_kv.second;
      for (int j = 0; j < n; ++j) {
        if (conj) {
          int l = k*n + j;
          int m = i*n + j;
          if (val != static_cast<T>(0)) {
            T prod = std::conj(val);
            prod *= static_cast<T>(alpha);
            y.data[l][m] += prod;
          }
        } else {
          int l = i*n + j;
          int m = k*n + j;
          if (val != static_cast<T>(0)) {
            T prod = val;
            prod *= alpha;
            y.data[l][m] += prod;
          }
        }
      }
    }
  }
}


template<typename T>
inline void kron_identity_left(const T& alpha,
                               const LilMatrix<T>& x,
                               const T& beta,
                               LilMatrix<T>& y,
                               bool conj=false) {
  if (beta == static_cast<T>(0)) {
    y.data.clear();
  } else {
    for (auto& y_ijv : y.data) {
      for (auto& y_jv: y_ijv.second) {
        y_jv.second *= beta;
      }
    }
  }

  int n = std::get<0>(x.shape);
  for (auto& A_kjv : x.data) {
    int k = A_kjv.first;
    for (auto& x_jv: A_kjv.second) {
      int j = x_jv.first;
      const T& val = x_jv.second;
      for (int i = 0; i < n; ++i) {
        if (conj) {
          int l = i*n + k;
          int m = i*n + j;
          if (val != static_cast<T>(0)) {
            T prod = std::conj(val);
            prod *= static_cast<T>(alpha);
            y.data[l][m] += prod;
          }
        } else {
          int l = i*n + j;
          int m = i*n + k;
          if (val != static_cast<T>(0)) {
            T prod = val;
            prod *= alpha;
            y.data[l][m] += prod;
          }
        }
      }
    }
  }
}

template<typename T>
struct AxpyImpl<T, LilMatrix> {
  static inline void func(T alpha,
                          const LilMatrix<T>& x,
                          LilMatrix<T>& y) {
    for (auto& x_ijv : x.data) {
      int i = x_ijv.first;
      for (auto& x_jv: x_ijv.second) {
        int j = x_jv.first;
        y.data[i][j] += alpha*x_jv.second;
      }
    }
  }
};


template<typename T>
inline void GemmLilMatrixImpl(T alpha,
                              const LilMatrix<T>& A,
                              const LilMatrix<T>& B,
                              T beta,
                              LilMatrix<T>& C) {
  if (beta == static_cast<T>(0)) {
    C.data.clear();
  } else {
    for (auto& C_ijv : C.data) {
      for (auto& C_jv: C_ijv.second) {
        C_jv.second *= beta;
      }
    }
  }
  
  for (auto& A_ikv : A.data) {
    int i = A_ikv.first;
    for (auto& A_kv: A_ikv.second) {
      int k = A_kv.first;
      try {
        for (auto& B_jv: B.data.at(k)) {
          int j  = B_jv.first;
          C.data[i][j] += alpha*A_kv.second*B_jv.second;
        }
      } catch (std::out_of_range&) {}
    }
  }
}

template<>
struct GemmImpl<complex64, LilMatrix, LilMatrix, LilMatrix> {
  static void func(complex64 alpha,
                   const LilMatrix<complex64>& A,
                   const LilMatrix<complex64>& B,
                   complex64 beta,
                   LilMatrix<complex64>& C) {
    GemmLilMatrixImpl(alpha, A, B, beta, C);
  }
};

template<>
struct GemmImpl<complex128, LilMatrix, LilMatrix, LilMatrix> {
  static void func(complex128 alpha,
                   const LilMatrix<complex128>& A,
                   const LilMatrix<complex128>& B,
                   complex128 beta,
                   LilMatrix<complex128>& C) {
    GemmLilMatrixImpl(alpha, A, B, beta, C);
  }
};

}
#endif /* LIL_MATRIX_H */
