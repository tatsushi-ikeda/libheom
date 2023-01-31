/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef LIBHEOM_DENSE_MATRIX_MKL_H
#define LIBHEOM_DENSE_MATRIX_MKL_H

#ifdef ENABLE_MKL

#ifdef __INTEL_COMPILER
#  include <aligned_new>
#endif
#include "const.h"
#include "env.h"
#include "linalg_engine/matrix_base.h"
#include "linalg_engine/lil_matrix.h"
#include "linalg_engine/dense_matrix.h"

#include "linalg_engine/include_mkl.h"
#include "linalg_engine/utility_mkl.h"

namespace libheom
{

template<order_t order>
constexpr MKL_LAYOUT mkl_layout = MKL_ROW_MAJOR;

template<>
constexpr MKL_LAYOUT mkl_layout<row_major> = MKL_ROW_MAJOR;
template<>
constexpr MKL_LAYOUT mkl_layout<col_major> = MKL_COL_MAJOR;

template<int num_level, typename dtype, order_t order>
class dense_matrix<num_level,dtype,order,mkl>
    : public matrix_base<num_level,dtype,order,mkl>
{
 public:
  std::tuple<int, int> shape;
  int major_stride;
  dtype* data;

  dense_matrix() : data(nullptr)
  {
    CALL_TRACE();
    this->major_stride = num_level;
  }

  ~dense_matrix()
  {
    CALL_TRACE();
    if (data != nullptr) {
      delete [] data;
    }
  }

  void set_shape(int rows, int cols)
  {
    CALL_TRACE();
    std::get<0>(shape) = rows;
    std::get<1>(shape) = cols;
    this->major_stride = std::get<shape_index<order>>(this->shape);
    if (this->data != nullptr) {
      delete [] this->data;
    }
    this->data = new (std::align_val_t{align_val<dtype>}) dtype [std::get<0>(this->shape)*std::get<1>(this->shape)];
    std::fill_n(this->data, std::get<0>(this->shape)*std::get<1>(this->shape), zero<dtype>());
  }

  void import(lil_matrix<dynamic,dtype,order,nil>& src)
  {
    CALL_TRACE();
    this->shape = src.shape;
    this->major_stride = std::get<shape_index<order>>(this->shape);

    if (data != nullptr) {
      delete [] data;
    }
    this->data = new (std::align_val_t{align_val<dtype>}) dtype [std::get<0>(this->shape)*std::get<1>(this->shape)];
    std::fill_n(this->data, std::get<0>(this->shape)*std::get<1>(this->shape), zero<dtype>());
    for (auto& data_ijv : src.data) {
      int i = data_ijv.first;
      for (auto& data_jv: data_ijv.second) {
        int j = data_jv.first;
        this->data[i*this->major_stride + j] = data_jv.second;
      }
    }
  }

  void dump(lil_matrix<dynamic,dtype,order,nil>& dest)
  {
    CALL_TRACE();
    int outer, inner;
    if constexpr (order == row_major) {
      outer = std::get<0>(this->shape);
      inner = std::get<1>(this->shape);
      dest.set_shape(std::get<0>(this->shape), std::get<1>(this->shape));
    } else {
      outer = std::get<1>(this->shape);
      inner = std::get<0>(this->shape);
      dest.set_shape(std::get<1>(this->shape), std::get<0>(this->shape));
    }

    for (int i = 0; i < outer; ++i) {
      for (int j = 0; j < inner; ++j) {
        dest.data[i][j] = this->data[i*this->major_stride + j];
      } 
    }
    dest.optimize();
  }
};

}

#endif

#endif
