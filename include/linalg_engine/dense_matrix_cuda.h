/* -*- mode:cuda -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef LIBHEOM_DENSE_MATRIX_CUDA_H
#define LIBHEOM_DENSE_MATRIX_CUDA_H

#ifdef ENABLE_CUDA

#include "env.h"
#include "env_gpu.h"

#include "linalg_engine/lil_matrix.h"
#include "linalg_engine/dense_matrix.h"

#include "linalg_engine/utility_cuda.h"

namespace libheom {

template<typename dtype, order_t order>
class dense_matrix<dynamic,dtype,order,cuda>
    : public matrix_base<dynamic,dtype,order,cuda>
{
 public:
  std::tuple<int, int> shape;
  int major_stride;
  device_t<dtype,env_gpu>* data_dev;

  dense_matrix() : data_dev(nullptr)
  {
    CALL_TRACE();
  }

  ~dense_matrix()
  {
    CALL_TRACE();
    if (data_dev != nullptr) {
      CUDA_CALL(cudaFree(data_dev));
    }
  }

  void set_shape(int rows, int cols)
  {
    CALL_TRACE();
    std::get<0>(shape) = rows;
    std::get<1>(shape) = cols;
    this->major_stride = std::get<shape_index<order>>(this->shape);
    if (data_dev != nullptr) {
      CUDA_CALL(cudaFree(data_dev));
    }
    CUDA_CALL(cudaMalloc(&this->data_dev, std::get<0>(this->shape)*std::get<1>(this->shape)*sizeof(dtype)));
    CUDA_CALL(cudaMemset(this->data_dev, 0, std::get<0>(this->shape)*std::get<1>(this->shape)*sizeof(dtype)));
  }
  
  void import(lil_matrix<dynamic,dtype,order,nil>& src)
  {
    CALL_TRACE();
    this->shape = src.shape;
    this->major_stride = std::get<shape_index<order>>(this->shape);

    dtype* data;

    data = new (std::align_val_t{align_val<dtype>}) dtype [std::get<0>(this->shape)*std::get<1>(this->shape)];
    std::fill_n(data, std::get<0>(this->shape)*std::get<1>(this->shape), zero<dtype>());
    for (auto& data_ijv : src.data) {
      int i = data_ijv.first;
      for (auto& data_jv: data_ijv.second) {
        int j = data_jv.first;
        data[i*this->major_stride + j] = data_jv.second;
      }
    }

    CUDA_CALL(cudaMalloc(&this->data_dev, std::get<0>(this->shape)*std::get<1>(this->shape)*sizeof(dtype)));
    CUDA_CALL(cudaMemcpy(this->data_dev, data, std::get<0>(this->shape)*std::get<1>(this->shape)*sizeof(dtype), cudaMemcpyHostToDevice));

    delete [] data;
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

    dtype* data;
    data = new (std::align_val_t{align_val<dtype>}) dtype [std::get<0>(this->shape)*std::get<1>(this->shape)];
    CUDA_CALL(cudaMemcpy(data, this->data_dev, std::get<0>(this->shape)*std::get<1>(this->shape)*sizeof(dtype), cudaMemcpyDeviceToHost));
    for (int i = 0; i < outer; ++i) {
      for (int j = 0; j < inner; ++j) {
        dest.data[i][j] = data[i*this->major_stride + j];
      } 
    }
    delete [] data;
    dest.optimize();
  }

};

}

#endif

#endif
