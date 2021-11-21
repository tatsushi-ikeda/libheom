/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef DENSE_MATRIX_GPU_H
#define DENSE_MATRIX_GPU_H

#include "dense_matrix.h"
#include "type_gpu.h"
#include "handle_gpu.h"

namespace libheom{

template <typename T>
class dense_matrix_gpu {
 public:
  std::tuple<int, int> shape;
  thrust::device_vector<GPU_TYPE(T)> data;
  
  inline dense_matrix_gpu<T>& operator = (const dense_matrix<T, Eigen::Dynamic>& rhs) {
    std::get<0>(shape) = rhs.rows();
    std::get<1>(shape) = rhs.cols();
    this->data.resize(std::get<0>(shape)*std::get<1>(shape));
    copy_vector_gpu(rhs.data(), this->data);
    return (*this);
  };

  explicit operator dense_matrix<T, Eigen::Dynamic> const () {
    dense_matrix<T, Eigen::Dynamic> out(std::get<0>(shape), std::get<1>(shape));
    copy_vector_gpu(this->data, out.data());
    return std::move(out);
  }

  RAW_GPU_TYPE(T)* data() {
    return raw_gpu_type_cast<T*>(data.data());
  }
  
  const RAW_GPU_TYPE(T)* data() const {
    return raw_gpu_type_cast<const T*>(data.data());
  }
};


template <typename T>
class dense_matrix_gpu_wrapper {
 public:
  std::tuple<int, int> shape;
  thrust::device_ptr<GPU_TYPE(T)> data;

  dense_matrix_gpu_wrapper(int shape_0, int shape_1,
                           thrust::device_ptr<GPU_TYPE(T)> data)
      : shape(shape_0, shape_1), data(data) {}

  RAW_GPU_TYPE(T)* data() {
    return raw_gpu_type_cast<T*>(data);
  }

  explicit operator dense_matrix<T, Eigen::Dynamic> const () {
    dense_matrix<T, Eigen::Dynamic> out(std::get<0>(shape), std::get<1>(shape));
    copy_vector_gpu(this->data(), out.data());
    return std::move(out);
  }

  const RAW_GPU_TYPE(T)* data() const {
    return raw_gpu_type_cast<const T*>(data);
  }
};


template <typename T>
class const_dense_matrix_gpu_wrapper {
 public:
  std::tuple<int, int> shape;
  const thrust::device_ptr<const GPU_TYPE(T)> data;

  const_dense_matrix_gpu_wrapper(int shape_0, int shape_1,
                             const thrust::device_ptr<const GPU_TYPE(T)> data)
      : shape(shape_0, shape_1), data(data) {}

  explicit operator dense_matrix<T, Eigen::Dynamic> const () {
    dense_matrix<T, Eigen::Dynamic> out(std::get<0>(shape), std::get<1>(shape));
    copy_vector_gpu(this->data(), out.data());
    return std::move(out);
  }
  
  const RAW_GPU_TYPE(T)* data() const {
    return raw_gpu_type_cast<const T*>(data);
  }

  
};

}
#endif /* DENSE_MATRIX_GPU_H */
