/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LICENSE.txt for licence.
 *------------------------------------------------------------------------*/

#include <gtest/gtest.h>
#include "libheom.h"

namespace {
using namespace libheom;
using c128 = complex128;
// mkl_spblas.h injects a global struct sparse_matrix; disambiguate explicitly.
using libheom::sparse_matrix;

class SparseMatrixFixture : public ::testing::Test {
protected:
    eigen eng;
};

// ---------------------------------------------------------------------------
// import: lil_matrix -> sparse_matrix
// ---------------------------------------------------------------------------

TEST_F(SparseMatrixFixture, ImportShape) {
    lil_matrix<dynamic, c128, row_major, nil> lil;
    lil.set_shape(3, 4);
    lil.push(0, 1, {1.0, 0.0});
    lil.push(2, 3, {2.0, 0.0});

    sparse_matrix<dynamic, c128, row_major, eigen> A;
    A.import(lil);

    EXPECT_EQ(std::get<0>(A.shape), 3);
    EXPECT_EQ(std::get<1>(A.shape), 4);
    EXPECT_EQ(A.data.nonZeros(), 2);
}

TEST_F(SparseMatrixFixture, ImportAndGemv) {
    // A = [[1, 2], [0, 3]], x = [4, 5]
    // y = A*x = [1*4+2*5, 0*4+3*5] = [14, 15]
    lil_matrix<dynamic, c128, row_major, nil> lil;
    lil.set_shape(2, 2);
    lil.push(0, 0, {1.0, 0.0});
    lil.push(0, 1, {2.0, 0.0});
    lil.push(1, 1, {3.0, 0.0});

    sparse_matrix<dynamic, c128, row_major, eigen> A;
    A.import(lil);

    c128 x[2] = {{4.0, 0.0}, {5.0, 0.0}};
    c128 y[2] = {};

    gemv<dynamic>(&eng, c128{1.0,0}, A, x, c128{0.0,0}, y, 2);

    EXPECT_DOUBLE_EQ(y[0].real(), 14.0);
    EXPECT_DOUBLE_EQ(y[1].real(), 15.0);
}

TEST_F(SparseMatrixFixture, GemvBetaAccumulates) {
    // y = alpha * A*x + beta * y_init
    // A = I (identity), x = [1, 2], y_init = [3, 4], alpha=2, beta=1
    // y = 2*[1,2] + 1*[3,4] = [5, 8]
    lil_matrix<dynamic, c128, row_major, nil> lil;
    lil.set_identity(2);

    sparse_matrix<dynamic, c128, row_major, eigen> A;
    A.import(lil);

    c128 x[2] = {{1.0, 0.0}, {2.0, 0.0}};
    c128 y[2] = {{3.0, 0.0}, {4.0, 0.0}};

    gemv<dynamic>(&eng, c128{2.0,0}, A, x, c128{1.0,0}, y, 2);

    EXPECT_DOUBLE_EQ(y[0].real(), 5.0);
    EXPECT_DOUBLE_EQ(y[1].real(), 8.0);
}

TEST_F(SparseMatrixFixture, GemvComplexMatrix) {
    // A = [[i, 0], [0, -i]], x = [1+i, 1-i]
    // A*x = [i*(1+i), -i*(1-i)] = [i+i^2 , -i+i^2] = [-1+i, -1-i]
    lil_matrix<dynamic, c128, row_major, nil> lil;
    lil.set_shape(2, 2);
    lil.push(0, 0, {0.0,  1.0}); //  i
    lil.push(1, 1, {0.0, -1.0}); // -i

    sparse_matrix<dynamic, c128, row_major, eigen> A;
    A.import(lil);

    c128 x[2] = {{1.0, 1.0}, {1.0, -1.0}};
    c128 y[2] = {};

    gemv<dynamic>(&eng, c128{1.0,0}, A, x, c128{0.0,0}, y, 2);

    EXPECT_NEAR(y[0].real(), -1.0, 1e-14);
    EXPECT_NEAR(y[0].imag(),  1.0, 1e-14);
    EXPECT_NEAR(y[1].real(), -1.0, 1e-14);
    EXPECT_NEAR(y[1].imag(), -1.0, 1e-14);
}

} // namespace
