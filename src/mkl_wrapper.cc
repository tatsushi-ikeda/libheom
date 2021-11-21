/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#include "mkl_wrapper.h"

namespace libheom {

template <>
void csr_matrix<complex64>::aux::init(csr_matrix<complex64>& mat) {
#if __INTEL_MKL__ >= 12
  mkl_sparse_c_create_csr(&handle,
                          SPARSE_INDEX_BASE_ZERO,
                          std::get<0>(mat.shape),
                          std::get<1>(mat.shape),
                          const_cast<int*>(mat.indptrb.data()),
                          const_cast<int*>(mat.indptre.data()),
                          const_cast<int*>(mat.indices.data()),
                          const_cast<complex64*>(mat.data.data()));
#endif
}


template <>
void csr_matrix<complex128>::aux::init(csr_matrix<complex128>& mat) {
#if __INTEL_MKL__ >= 12
  mkl_sparse_z_create_csr(&handle,
                          SPARSE_INDEX_BASE_ZERO,
                          std::get<0>(mat.shape),
                          std::get<1>(mat.shape),
                          const_cast<int*>(mat.indptrb.data()),
                          const_cast<int*>(mat.indptre.data()),
                          const_cast<int*>(mat.indices.data()),
                          const_cast<complex128*>(mat.data.data()));
#endif
}

template<typename T>
csr_matrix<T>::~csr_matrix() {
  fin_aux();
};

}

namespace libheom {
template class csr_matrix<complex64>;
template class csr_matrix<complex128>;
}
