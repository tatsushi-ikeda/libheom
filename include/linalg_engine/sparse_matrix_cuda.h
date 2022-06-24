/* -*- mode:cuda -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef SPARSE_MATRIX_CUDA_H
#define SPARSE_MATRIX_CUDA_H

#ifdef ENABLE_CUDA

#include "const.h"
#include "env.h"

#include "linalg_engine/matrix_base.h"
#include "linalg_engine/lil_matrix.h"
#include "linalg_engine/sparse_matrix.h"

#include "linalg_engine/linalg_engine.h"

#include "linalg_engine/include_cuda.h"
#include "linalg_engine/utility_cuda.h"

namespace libheom
{

template<typename dtype>
constexpr cudaDataType_t cuda_type_const = CUDA_R_32F;

template <>
constexpr cudaDataType_t cuda_type_const<float32>    = CUDA_R_32F;
template <>
constexpr cudaDataType_t cuda_type_const<complex64>  = CUDA_C_32F;
template <>
constexpr cudaDataType_t cuda_type_const<float64>    = CUDA_R_64F;
template <>
constexpr cudaDataType_t cuda_type_const<complex128> = CUDA_C_64F;


template<bool order>
constexpr cusparseOrder_t cusparse_order = CUSPARSE_ORDER_ROW;

template<>
constexpr cusparseOrder_t cusparse_order<row_major> = CUSPARSE_ORDER_ROW;
template<>
constexpr cusparseOrder_t cusparse_order<col_major> = CUSPARSE_ORDER_COL;


template<bool order>
constexpr cusparseOrder_t cusparse_order_t = CUSPARSE_ORDER_COL;

template<>
constexpr cusparseOrder_t cusparse_order_t<row_major> = CUSPARSE_ORDER_COL;
template<>
constexpr cusparseOrder_t cusparse_order_t<col_major> = CUSPARSE_ORDER_ROW;


template<bool order>
constexpr cusparseSpMMAlg_t cusparse_alg = CUSPARSE_SPMM_CSR_ALG2;

template<>
constexpr cusparseSpMMAlg_t cusparse_alg<row_major> = CUSPARSE_SPMM_CSR_ALG2;
template<>
constexpr cusparseSpMMAlg_t cusparse_alg<col_major> = CUSPARSE_SPMM_CSR_ALG1;

template<bool order>
constexpr cusparseSpMMAlg_t cusparse_alg_t = CUSPARSE_SPMM_CSR_ALG1;

template<>
constexpr cusparseSpMMAlg_t cusparse_alg_t<row_major> = CUSPARSE_SPMM_CSR_ALG1;
template<>
constexpr cusparseSpMMAlg_t cusparse_alg_t<col_major> = CUSPARSE_SPMM_CSR_ALG2;

template<bool order>
inline cusparseStatus_t cusparse_create(cusparseSpMatDescr_t* dsc,
                                        int64_t rows, int64_t cols, int64_t nnz,
                                        void* outer, void* inner, void* data,
                                        cudaDataType_t dtype_const);

template<>
inline cusparseStatus_t cusparse_create<row_major>(cusparseSpMatDescr_t* dsc,
                                                   int64_t rows, int64_t cols, int64_t nnz,
                                                   void* outer, void* inner, void* data,
                                                   cudaDataType_t dtype_const)
{
  return cusparseCreateCsr(dsc, rows, cols, nnz,
                           outer, inner, data,
                           (sizeof(int) == 8) ? CUSPARSE_INDEX_64I : CUSPARSE_INDEX_32I,
                           (sizeof(int) == 8) ? CUSPARSE_INDEX_64I : CUSPARSE_INDEX_32I,
                           CUSPARSE_INDEX_BASE_ZERO,
                           dtype_const);
}

template<>
inline cusparseStatus_t cusparse_create<col_major>(cusparseSpMatDescr_t* dsc,
                                                   int64_t rows, int64_t cols, int64_t nnz,
                                                   void* outer, void* inner, void* data,
                                                   cudaDataType_t dtype_const)
{
  return cusparseCreateCsc(dsc, rows, cols, nnz,
                           outer, inner, data,
                           (sizeof(int) == 8) ? CUSPARSE_INDEX_64I : CUSPARSE_INDEX_32I,
                           (sizeof(int) == 8) ? CUSPARSE_INDEX_64I : CUSPARSE_INDEX_32I,
                           CUSPARSE_INDEX_BASE_ZERO,
                           dtype_const);
}

template<typename dtype, bool order>
class sparse_matrix<dynamic,dtype,order,cuda>
    : public matrix_base<dynamic,dtype,order,cuda>
{
 public:
  std::tuple<int, int> shape;
  int major_stride;
  const int base = 0;
  device_t<dtype,env_gpu>* data_dev;
  device_t<int,env_gpu>*   inner_dev;
  device_t<int,env_gpu>*   outer_dev;
  cusparseSpMatDescr_t     dsc;


  sparse_matrix() : data_dev(nullptr), inner_dev(nullptr), outer_dev(nullptr)
  {
    CALL_TRACE();
  }

  ~sparse_matrix()
  {
    CALL_TRACE();
    if (data_dev != nullptr) {
      CUDA_CALL(cudaFree(data_dev));
      CUDA_CALL(cudaFree(inner_dev));
      CUDA_CALL(cudaFree(outer_dev));
      CUSPARSE_CALL(cusparseDestroySpMat(dsc));
    }
  }

  void import(lil_matrix<dynamic,dtype,order,nil>& src)
  {
    CALL_TRACE();
    this->shape = src.shape;
    this->major_stride = std::get<shape_index<order>>(this->shape);
    int n_outer        = std::get<shape_index<order>^1>(this->shape);

    std::vector<dtype>  data;
    std::vector<int>    inner;
    std::vector<int>    outer_b(n_outer+1);
    std::vector<int>    outer_e(n_outer);

    int ptr = base;
    int outer_old = -1;

    // pointer_b -> outer_b;
    // pointer_e -> outer_e;
    // values -> data;
    // inner -> columns;
    // row_old -> outer_old

    for (auto& data_ijv : src.data) {
      int i = data_ijv.first;
      for (auto& data_jv: data_ijv.second) {
        int j = data_jv.first;

        if (i != outer_old) {
          if (outer_old != -1) {
            outer_e[outer_old] = ptr;
          }
          ++outer_old;
          while (outer_old < i) {
            outer_b[outer_old] = ptr;
            outer_e[outer_old] = ptr;
            ++outer_old;
          }
          outer_b[i] = ptr;
        }
        data.push_back(data_jv.second);
        inner.push_back(j + base);
        ++ptr;
      }
    }

    if (outer_old != -1) {
      outer_e[outer_old] = ptr;
    }
    int i = this->major_stride;
    if (i != outer_old) {
      ++outer_old;
      while (outer_old < i) {
        outer_b[outer_old] = ptr;
        outer_e[outer_old] = ptr;
        ++outer_old;
      }
    }
    outer_b[this->major_stride] = data.size() + base;

    CUDA_CALL(cudaMalloc(&this->data_dev, data.size()*sizeof(dtype)));
    CUDA_CALL(cudaMemcpy(this->data_dev, &data[0],     data.size()*sizeof(dtype),  cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&this->inner_dev, inner.size()*sizeof(int)));
    CUDA_CALL(cudaMemcpy(this->inner_dev, &inner[0],   inner.size()*sizeof(int),   cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&this->outer_dev, outer_b.size()*sizeof(int)));
    CUDA_CALL(cudaMemcpy(this->outer_dev, &outer_b[0], outer_b.size()*sizeof(int), cudaMemcpyHostToDevice));

    CUSPARSE_CALL((cusparse_create<order>(&this->dsc,
                                          std::get<0>(this->shape), std::get<1>(this->shape), data.size(),
                                          this->outer_dev, this->inner_dev, this->data_dev,
                                          cuda_type_const<dtype>)));
  }
};

}

#endif

#endif /* SPARSE_MATRIX_CUDA_H */
