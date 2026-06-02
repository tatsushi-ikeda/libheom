/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LICENSE.txt for license.
 *------------------------------------------------------------------------*/

#include <gtest/gtest.h>
#include "libheom.h"

namespace {
using namespace libheom;
using c128 = complex128;

static constexpr double EPS = 1e-14;

class MklEngineFixture : public ::testing::Test {
protected:
    mkl eng;
};

// ---------------------------------------------------------------------------
// axpy: y += a * x
// ---------------------------------------------------------------------------

TEST_F(MklEngineFixture, Axpy_Real) {
    c128 x[3] = {{1.0,0},{2.0,0},{3.0,0}};
    c128 y[3] = {{4.0,0},{5.0,0},{6.0,0}};
    axpy<dynamic>(&eng, c128{2.0,0}, x, y, 3);
    EXPECT_NEAR(y[0].real(),  6.0, EPS);
    EXPECT_NEAR(y[1].real(),  9.0, EPS);
    EXPECT_NEAR(y[2].real(), 12.0, EPS);
}

TEST_F(MklEngineFixture, Axpy_Complex) {
    c128 x[2] = {{1.0, 2.0}, {3.0, 4.0}};
    c128 y[2] = {{0.0, 0.0}, {0.0, 0.0}};
    axpy<dynamic>(&eng, c128{0.0, 1.0}, x, y, 2); // y += i * x
    EXPECT_NEAR(y[0].real(), -2.0, EPS);
    EXPECT_NEAR(y[0].imag(),  1.0, EPS);
    EXPECT_NEAR(y[1].real(), -4.0, EPS);
    EXPECT_NEAR(y[1].imag(),  3.0, EPS);
}

TEST_F(MklEngineFixture, Axpy_ZeroAlpha) {
    c128 x[2] = {{99.0,0},{99.0,0}};
    c128 y[2] = {{1.0, 0},{2.0, 0}};
    axpy<dynamic>(&eng, c128{0.0,0}, x, y, 2);
    EXPECT_NEAR(y[0].real(), 1.0, EPS);
    EXPECT_NEAR(y[1].real(), 2.0, EPS);
}

// ---------------------------------------------------------------------------
// scal: y = a * y
// ---------------------------------------------------------------------------

TEST_F(MklEngineFixture, Scal_Real) {
    c128 y[3] = {{1.0,0},{2.0,0},{3.0,0}};
    scal<dynamic>(&eng, c128{3.0,0}, y, 3);
    EXPECT_NEAR(y[0].real(), 3.0, EPS);
    EXPECT_NEAR(y[1].real(), 6.0, EPS);
    EXPECT_NEAR(y[2].real(), 9.0, EPS);
}

TEST_F(MklEngineFixture, Scal_Zero) {
    c128 y[2] = {{5.0,1.0},{3.0,2.0}};
    scal<dynamic>(&eng, c128{0.0,0}, y, 2);
    EXPECT_NEAR(y[0].real(), 0.0, EPS);
    EXPECT_NEAR(y[0].imag(), 0.0, EPS);
    EXPECT_NEAR(y[1].real(), 0.0, EPS);
    EXPECT_NEAR(y[1].imag(), 0.0, EPS);
}

TEST_F(MklEngineFixture, Scal_Complex) {
    c128 y[1] = {{1.0, 0.0}};
    scal<dynamic>(&eng, c128{0.0, 1.0}, y, 1); // y = i * 1 = i
    EXPECT_NEAR(y[0].real(), 0.0, EPS);
    EXPECT_NEAR(y[0].imag(), 1.0, EPS);
}

// ---------------------------------------------------------------------------
// gemv: y = alpha * A * x + beta * y
// ---------------------------------------------------------------------------

TEST_F(MklEngineFixture, Gemv_Dense) {
    // A = [[2, 0], [0, 3]], x = [1, 1], y = A*x = [2, 3]
    lil_matrix<dynamic, c128, row_major, nil> lil;
    lil.set_shape(2, 2);
    lil.push(0, 0, {2.0, 0.0});
    lil.push(1, 1, {3.0, 0.0});

    dense_matrix<dynamic, c128, row_major, mkl> A;
    A.set_shape(2, 2);
    A.import(lil);

    c128 x[2] = {{1.0,0},{1.0,0}};
    c128 y[2] = {};

    gemv<dynamic>(&eng, c128{1.0,0}, A, x, c128{0.0,0}, y, 2);

    EXPECT_NEAR(y[0].real(), 2.0, EPS);
    EXPECT_NEAR(y[1].real(), 3.0, EPS);
}

TEST_F(MklEngineFixture, Gemv_BetaAccumulate) {
    // y = 1*I*x + 2*y_0, x=[1,1], y_0=[3,4] -> y=[7,9]
    lil_matrix<dynamic, c128, row_major, nil> lil;
    lil.set_identity(2);
    dense_matrix<dynamic, c128, row_major, mkl> I;
    I.set_shape(2, 2); I.import(lil);

    c128 x[2] = {{1.0,0},{1.0,0}};
    c128 y[2] = {{3.0,0},{4.0,0}};
    gemv<dynamic>(&eng, c128{1.0,0}, I, x, c128{2.0,0}, y, 2);

    EXPECT_NEAR(y[0].real(), 7.0, EPS);
    EXPECT_NEAR(y[1].real(), 9.0, EPS);
}

// ---------------------------------------------------------------------------
// gemm: C = alpha * A * B + beta * C
// A=[[1,2],[3,4]], B=[[5,6],[7,8]] -> A*B=[[19,22],[43,50]]
// ---------------------------------------------------------------------------

TEST_F(MklEngineFixture, Gemm_2x2) {
    lil_matrix<dynamic, c128, row_major, nil> lil;
    lil.set_shape(2, 2);
    lil.push(0, 0, {1.0,0}); lil.push(0, 1, {2.0,0});
    lil.push(1, 0, {3.0,0}); lil.push(1, 1, {4.0,0});

    dense_matrix<dynamic, c128, row_major, mkl> A;
    A.set_shape(2, 2); A.import(lil);

    c128 B[4] = {{5.0,0},{6.0,0},{7.0,0},{8.0,0}};
    c128 C[4] = {};

    gemm<dynamic>(&eng, c128{1.0,0}, A, B, c128{0.0,0}, C, 2);

    EXPECT_NEAR(C[0].real(), 19.0, EPS);
    EXPECT_NEAR(C[1].real(), 22.0, EPS);
    EXPECT_NEAR(C[2].real(), 43.0, EPS);
    EXPECT_NEAR(C[3].real(), 50.0, EPS);
}

TEST_F(MklEngineFixture, Gemm_AlphaBeta) {
    // C = 0.5 * I*I + 3 * C_0 with C_0 = I -> C = 3.5*I
    lil_matrix<dynamic, c128, row_major, nil> lil;
    lil.set_identity(2);
    dense_matrix<dynamic, c128, row_major, mkl> I;
    I.set_shape(2, 2); I.import(lil);

    c128 B[4] = {{1.0,0},{0.0,0},{0.0,0},{1.0,0}};
    c128 C[4] = {{1.0,0},{0.0,0},{0.0,0},{1.0,0}};

    gemm<dynamic>(&eng, c128{0.5,0}, I, B, c128{3.0,0}, C, 2);

    EXPECT_NEAR(C[0].real(), 3.5, EPS);
    EXPECT_NEAR(C[1].real(), 0.0, EPS);
    EXPECT_NEAR(C[2].real(), 0.0, EPS);
    EXPECT_NEAR(C[3].real(), 3.5, EPS);
}

} // namespace
