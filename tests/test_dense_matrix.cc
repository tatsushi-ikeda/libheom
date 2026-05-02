/* -*- mode:c++ -*-
 * LibHEOM -- unit tests for dense_matrix (Eigen backend)
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LICENSE.txt for licence.
 *------------------------------------------------------------------------*/

#include <gtest/gtest.h>
#include "libheom.h"

namespace {
using namespace libheom;
using c128 = complex128;

class DenseMatrixFixture : public ::testing::Test {
protected:
    eigen eng;
};

// ---------------------------------------------------------------------------
// import: lil_matrix -> dense_matrix
// ---------------------------------------------------------------------------

TEST_F(DenseMatrixFixture, ImportValues) {
    lil_matrix<dynamic, c128, row_major, nil> lil;
    lil.set_shape(2, 2);
    lil.push(0, 0, {1.0, 0.0});
    lil.push(0, 1, {2.0, 0.0});
    lil.push(1, 0, {3.0, 0.0});
    lil.push(1, 1, {4.0, 0.0});

    dense_matrix<dynamic, c128, row_major, eigen> A;
    A.set_shape(2, 2);
    A.import(lil);

    EXPECT_DOUBLE_EQ(A.data(0, 0).real(), 1.0);
    EXPECT_DOUBLE_EQ(A.data(0, 1).real(), 2.0);
    EXPECT_DOUBLE_EQ(A.data(1, 0).real(), 3.0);
    EXPECT_DOUBLE_EQ(A.data(1, 1).real(), 4.0);
}

TEST_F(DenseMatrixFixture, ImportZeroFillsRemainder) {
    lil_matrix<dynamic, c128, row_major, nil> lil;
    lil.set_shape(2, 2);
    lil.push(0, 0, {5.0, 0.0}); // only one entry

    dense_matrix<dynamic, c128, row_major, eigen> A;
    A.set_shape(2, 2);
    A.import(lil);

    EXPECT_DOUBLE_EQ(A.data(0, 0).real(), 5.0);
    EXPECT_DOUBLE_EQ(A.data(0, 1).real(), 0.0);
    EXPECT_DOUBLE_EQ(A.data(1, 0).real(), 0.0);
    EXPECT_DOUBLE_EQ(A.data(1, 1).real(), 0.0);
}

// ---------------------------------------------------------------------------
// dump: dense_matrix -> lil_matrix (roundtrip)
// ---------------------------------------------------------------------------

TEST_F(DenseMatrixFixture, DumpRoundtrip) {
    lil_matrix<dynamic, c128, row_major, nil> lil_src, lil_out;
    lil_src.set_shape(2, 2);
    lil_src.push(0, 0, {1.0,  2.0});
    lil_src.push(0, 1, {3.0, -1.0});
    lil_src.push(1, 0, {0.0,  4.0});
    lil_src.push(1, 1, {5.0,  0.0});

    dense_matrix<dynamic, c128, row_major, eigen> A;
    A.set_shape(2, 2);
    A.import(lil_src);
    A.dump(lil_out);

    EXPECT_EQ(lil_out.data.at(0).at(0), c128(1.0,  2.0));
    EXPECT_EQ(lil_out.data.at(0).at(1), c128(3.0, -1.0));
    EXPECT_EQ(lil_out.data.at(1).at(0), c128(0.0,  4.0));
    EXPECT_EQ(lil_out.data.at(1).at(1), c128(5.0,  0.0));
}

// ---------------------------------------------------------------------------
// gemm: C = alpha*A*B + beta*C
// Analytic check: A=[[1,2],[3,4]], B=[[5,6],[7,8]] -> A*B=[[19,22],[43,50]]
// ---------------------------------------------------------------------------

TEST_F(DenseMatrixFixture, Gemm2x2) {
    lil_matrix<dynamic, c128, row_major, nil> lil;
    lil.set_shape(2, 2);
    lil.push(0, 0, {1.0, 0.0}); lil.push(0, 1, {2.0, 0.0});
    lil.push(1, 0, {3.0, 0.0}); lil.push(1, 1, {4.0, 0.0});

    dense_matrix<dynamic, c128, row_major, eigen> A;
    A.set_shape(2, 2);
    A.import(lil);

    c128 B[4] = {{5.0,0},{6.0,0},{7.0,0},{8.0,0}};
    c128 C[4] = {};

    gemm<dynamic>(&eng, c128{1.0,0}, A, B, c128{0.0,0}, C, 2);

    EXPECT_DOUBLE_EQ(C[0].real(), 19.0);
    EXPECT_DOUBLE_EQ(C[1].real(), 22.0);
    EXPECT_DOUBLE_EQ(C[2].real(), 43.0);
    EXPECT_DOUBLE_EQ(C[3].real(), 50.0);
}

TEST_F(DenseMatrixFixture, GemmBetaAccumulates) {
    // C = 1*I*I + 2*I = 3*I (with I = identity 2x2)
    lil_matrix<dynamic, c128, row_major, nil> lil;
    lil.set_identity(2);

    dense_matrix<dynamic, c128, row_major, eigen> I;
    I.set_shape(2, 2);
    I.import(lil);

    c128 B[4] = {{1.0,0},{0.0,0},{0.0,0},{1.0,0}}; // identity
    c128 C[4] = {{1.0,0},{0.0,0},{0.0,0},{1.0,0}}; // start with I

    gemm<dynamic>(&eng, c128{1.0,0}, I, B, c128{2.0,0}, C, 2);

    // C = 1*I*I + 2*I = I + 2I = 3I
    EXPECT_DOUBLE_EQ(C[0].real(), 3.0);
    EXPECT_DOUBLE_EQ(C[1].real(), 0.0);
    EXPECT_DOUBLE_EQ(C[2].real(), 0.0);
    EXPECT_DOUBLE_EQ(C[3].real(), 3.0);
}

} // namespace
