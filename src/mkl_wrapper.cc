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
void CsrMatrix<complex64>::Aux::Init(CsrMatrix<complex64>& mat) {
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
void CsrMatrix<complex128>::Aux::Init(CsrMatrix<complex128>& mat) {
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
CsrMatrix<T>::~CsrMatrix() {
  FinAux();
};

}

namespace libheom {
template class CsrMatrix<complex64>;
template class CsrMatrix<complex128>;
}
