/* -*- mode:cuda -*-
 * LibHEOM -- unit tests for linalg_engine<cuda>: axpy, scal, gemv, gemm,
 *            sync_to_children / sync_from_children
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LICENSE.txt for licence.
 *------------------------------------------------------------------------*/

#include <gtest/gtest.h>
#include "libheom.h"

namespace {
using namespace libheom;
using c128     = complex128;
using env_gpu_t = engine_env<cuda>;

static constexpr double EPS = 1e-14;

// Helper: allocate GPU buffer, copy host data in, run f(d_ptrs...), copy out.
// Using RAII wrapper to avoid leaks on EXPECT failures.
template<typename dtype>
struct DevBuf {
    device_t<dtype, env_gpu_t>* ptr;
    int n;
    explicit DevBuf(int n_) : n(n_) { ptr = new_dev<dtype, env_gpu_t>(n); }
    DevBuf(const dtype* host, int n_) : n(n_) {
        ptr = new_dev<dtype, env_gpu_t>(n);
        host2dev<dtype, env_gpu_t>(const_cast<dtype*>(host), ptr, n);
    }
    ~DevBuf() { delete_dev<dtype, env_gpu_t>(ptr); }
    void to_host(dtype* host) { dev2host<dtype, env_gpu_t>(ptr, host, n); }
};

class CudaEngineFixture : public ::testing::Test {
protected:
    cuda eng{0};
};

// ---------------------------------------------------------------------------
// axpy: y += a * x
// ---------------------------------------------------------------------------

TEST_F(CudaEngineFixture, Axpy_Real) {
    c128 h_x[3] = {{1.0,0},{2.0,0},{3.0,0}};
    c128 h_y[3] = {{4.0,0},{5.0,0},{6.0,0}};
    DevBuf<c128> dx(h_x, 3), dy(h_y, 3);
    axpy<dynamic>(&eng, c128{2.0,0}, dx.ptr, dy.ptr, 3);
    cudaStreamSynchronize(eng.stream);
    dy.to_host(h_y);
    EXPECT_NEAR(h_y[0].real(),  6.0, EPS);
    EXPECT_NEAR(h_y[1].real(),  9.0, EPS);
    EXPECT_NEAR(h_y[2].real(), 12.0, EPS);
}

TEST_F(CudaEngineFixture, Axpy_Complex) {
    c128 h_x[2] = {{1.0, 2.0}, {3.0, 4.0}};
    c128 h_y[2] = {{0.0, 0.0}, {0.0, 0.0}};
    DevBuf<c128> dx(h_x, 2), dy(h_y, 2);
    axpy<dynamic>(&eng, c128{0.0, 1.0}, dx.ptr, dy.ptr, 2);
    cudaStreamSynchronize(eng.stream);
    dy.to_host(h_y);
    // i*(1+2i) = -2+i
    EXPECT_NEAR(h_y[0].real(), -2.0, EPS);
    EXPECT_NEAR(h_y[0].imag(),  1.0, EPS);
    // i*(3+4i) = -4+3i
    EXPECT_NEAR(h_y[1].real(), -4.0, EPS);
    EXPECT_NEAR(h_y[1].imag(),  3.0, EPS);
}

TEST_F(CudaEngineFixture, Axpy_ZeroAlpha) {
    c128 h_x[2] = {{99.0,0},{99.0,0}};
    c128 h_y[2] = {{1.0, 0},{2.0, 0}};
    DevBuf<c128> dx(h_x, 2), dy(h_y, 2);
    axpy<dynamic>(&eng, c128{0.0,0}, dx.ptr, dy.ptr, 2);
    cudaStreamSynchronize(eng.stream);
    dy.to_host(h_y);
    EXPECT_NEAR(h_y[0].real(), 1.0, EPS);
    EXPECT_NEAR(h_y[1].real(), 2.0, EPS);
}

// ---------------------------------------------------------------------------
// scal: y = a * y
// ---------------------------------------------------------------------------

TEST_F(CudaEngineFixture, Scal_Real) {
    c128 h_y[3] = {{1.0,0},{2.0,0},{3.0,0}};
    DevBuf<c128> dy(h_y, 3);
    scal<dynamic>(&eng, c128{3.0,0}, dy.ptr, 3);
    cudaStreamSynchronize(eng.stream);
    dy.to_host(h_y);
    EXPECT_NEAR(h_y[0].real(), 3.0, EPS);
    EXPECT_NEAR(h_y[1].real(), 6.0, EPS);
    EXPECT_NEAR(h_y[2].real(), 9.0, EPS);
}

TEST_F(CudaEngineFixture, Scal_Zero) {
    c128 h_y[2] = {{5.0,1.0},{3.0,2.0}};
    DevBuf<c128> dy(h_y, 2);
    scal<dynamic>(&eng, c128{0.0,0}, dy.ptr, 2);
    cudaStreamSynchronize(eng.stream);
    dy.to_host(h_y);
    EXPECT_NEAR(h_y[0].real(), 0.0, EPS);
    EXPECT_NEAR(h_y[0].imag(), 0.0, EPS);
    EXPECT_NEAR(h_y[1].real(), 0.0, EPS);
    EXPECT_NEAR(h_y[1].imag(), 0.0, EPS);
}

TEST_F(CudaEngineFixture, Scal_Complex) {
    c128 h_y[1] = {{1.0, 0.0}};
    DevBuf<c128> dy(h_y, 1);
    scal<dynamic>(&eng, c128{0.0, 1.0}, dy.ptr, 1); // y = i
    cudaStreamSynchronize(eng.stream);
    dy.to_host(h_y);
    EXPECT_NEAR(h_y[0].real(), 0.0, EPS);
    EXPECT_NEAR(h_y[0].imag(), 1.0, EPS);
}

// ---------------------------------------------------------------------------
// gemv: y = alpha * A * x + beta * y  (dense_matrix on GPU)
// ---------------------------------------------------------------------------

TEST_F(CudaEngineFixture, Gemv_Dense) {
    // A = [[2,0],[0,3]], x=[1,1], y=A*x=[2,3]
    lil_matrix<dynamic, c128, row_major, nil> lil;
    lil.set_shape(2, 2);
    lil.push(0, 0, {2.0, 0.0});
    lil.push(1, 1, {3.0, 0.0});

    dense_matrix<dynamic, c128, row_major, cuda> A;
    A.import(lil);

    c128 h_x[2] = {{1.0,0},{1.0,0}};
    c128 h_y[2] = {{0.0,0},{0.0,0}};
    DevBuf<c128> dx(h_x, 2), dy(h_y, 2);

    gemv<dynamic>(&eng, c128{1.0,0}, A, dx.ptr, c128{0.0,0}, dy.ptr, 2);
    cudaStreamSynchronize(eng.stream);
    dy.to_host(h_y);

    EXPECT_NEAR(h_y[0].real(), 2.0, EPS);
    EXPECT_NEAR(h_y[1].real(), 3.0, EPS);
}

TEST_F(CudaEngineFixture, Gemv_BetaAccumulate) {
    // y = 1*I*x + 2*y_0, x=[1,1], y_0=[3,4] -> y=[7,9]
    lil_matrix<dynamic, c128, row_major, nil> lil;
    lil.set_identity(2);
    dense_matrix<dynamic, c128, row_major, cuda> I;
    I.import(lil);

    c128 h_x[2] = {{1.0,0},{1.0,0}};
    c128 h_y[2] = {{3.0,0},{4.0,0}};
    DevBuf<c128> dx(h_x, 2), dy(h_y, 2);

    gemv<dynamic>(&eng, c128{1.0,0}, I, dx.ptr, c128{2.0,0}, dy.ptr, 2);
    cudaStreamSynchronize(eng.stream);
    dy.to_host(h_y);

    EXPECT_NEAR(h_y[0].real(), 7.0, EPS);
    EXPECT_NEAR(h_y[1].real(), 9.0, EPS);
}

// ---------------------------------------------------------------------------
// gemm: C = alpha * A * B + beta * C
// A=[[1,2],[3,4]], B=[[5,6],[7,8]] -> A*B=[[19,22],[43,50]]
// ---------------------------------------------------------------------------

TEST_F(CudaEngineFixture, Gemm_2x2) {
    lil_matrix<dynamic, c128, row_major, nil> lil;
    lil.set_shape(2, 2);
    lil.push(0, 0, {1.0,0}); lil.push(0, 1, {2.0,0});
    lil.push(1, 0, {3.0,0}); lil.push(1, 1, {4.0,0});

    dense_matrix<dynamic, c128, row_major, cuda> A;
    A.import(lil);

    c128 h_B[4] = {{5.0,0},{6.0,0},{7.0,0},{8.0,0}};
    c128 h_C[4] = {{0.0,0},{0.0,0},{0.0,0},{0.0,0}};
    DevBuf<c128> dB(h_B, 4), dC(h_C, 4);

    gemm<dynamic>(&eng, c128{1.0,0}, A, dB.ptr, c128{0.0,0}, dC.ptr, 2);
    cudaStreamSynchronize(eng.stream);
    dC.to_host(h_C);

    EXPECT_NEAR(h_C[0].real(), 19.0, EPS);
    EXPECT_NEAR(h_C[1].real(), 22.0, EPS);
    EXPECT_NEAR(h_C[2].real(), 43.0, EPS);
    EXPECT_NEAR(h_C[3].real(), 50.0, EPS);
}

TEST_F(CudaEngineFixture, Gemm_AlphaBeta) {
    // C = 0.5*I*I + 3*C_0 with C_0=I -> C = 3.5*I
    lil_matrix<dynamic, c128, row_major, nil> lil;
    lil.set_identity(2);
    dense_matrix<dynamic, c128, row_major, cuda> I;
    I.import(lil);

    c128 h_B[4] = {{1.0,0},{0.0,0},{0.0,0},{1.0,0}};
    c128 h_C[4] = {{1.0,0},{0.0,0},{0.0,0},{1.0,0}};
    DevBuf<c128> dB(h_B, 4), dC(h_C, 4);

    gemm<dynamic>(&eng, c128{0.5,0}, I, dB.ptr, c128{3.0,0}, dC.ptr, 2);
    cudaStreamSynchronize(eng.stream);
    dC.to_host(h_C);

    EXPECT_NEAR(h_C[0].real(), 3.5, EPS);
    EXPECT_NEAR(h_C[1].real(), 0.0, EPS);
    EXPECT_NEAR(h_C[2].real(), 0.0, EPS);
    EXPECT_NEAR(h_C[3].real(), 3.5, EPS);
}

// ---------------------------------------------------------------------------
// sync_to_children / sync_from_children: parent enqueues work; child should
// see it after sync_to_children, and vice versa.
// We verify functional correctness: child axpy + sync_from_children produces
// the right result visible to the parent stream.
// ---------------------------------------------------------------------------

TEST_F(CudaEngineFixture, SyncChildrenRoundtrip) {
    // Parent initialises y=[0,0] on GPU.
    // Child performs axpy: y += 1*x with x=[3,7].
    // After sync_from_children, parent reads y and asserts y=[3,7].
    eng.create_children(1);
    auto* child = static_cast<cuda*>(eng.get_child(0));

    c128 h_x[2] = {{3.0,0},{7.0,0}};
    c128 h_y[2] = {{0.0,0},{0.0,0}};
    DevBuf<c128> dx(h_x, 2), dy(h_y, 2);

    // Parent initialises dx on its stream; child must not start before this.
    eng.sync_to_children();

    axpy<dynamic>(child, c128{1.0,0}, dx.ptr, dy.ptr, 2);

    eng.sync_from_children();

    // Parent reads dy on its own stream after sync.
    dy.to_host(h_y);
    cudaStreamSynchronize(eng.stream);

    EXPECT_NEAR(h_y[0].real(), 3.0, EPS);
    EXPECT_NEAR(h_y[1].real(), 7.0, EPS);
}

} // namespace
