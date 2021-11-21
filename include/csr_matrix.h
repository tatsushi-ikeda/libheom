/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

#include <assert.h>

#include <memory>
#include <vector>
#include <tuple>
#include <ostream>
#include <fstream>

#include <Eigen/Sparse>

#include "macro.h"
#include "printer.h"

namespace libheom {

template<typename T,
         int N>
using csr_matrix = Eigen::SparseMatrix<T, Eigen::RowMajor>;

}

#endif /* CSR_MATRIX_H */
