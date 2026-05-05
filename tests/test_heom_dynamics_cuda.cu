/* -*- mode:cuda -*-
 * LibHEOM -- GoogleTest: CUDA heom_hilb/heom_liou/heom_ado dynamics vs Eigen
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LICENSE.txt for licence.
 *
 * Verifies that calc_diff_impl with the CUDA engine produces results
 * numerically consistent with the Eigen engine for heom_liou, heom_hilb,
 * and heom_ado.
 *
 * System: n_level=2, H=diag(1,0), V=Pauli-X
 *   Bath: 1 noise, 1 Matsubara pole
 *         gamma=-1.0, phi_0=1.0, sigma=0.5
 *         S=[[0.3]], A=[[0.1]], s_delta=0.0
 *   truncation_depth=2 -> n_hierarchy=3, main_size=12
 *------------------------------------------------------------------------*/

#include <gtest/gtest.h>
#include "libheom.h"
#include <vector>

namespace {
using namespace libheom;
using c128       = complex128;
using env_gpu_t  = engine_env<cuda>;

static constexpr double EPS       = 1e-12;
static constexpr int    N_LEVEL   = 2;
static constexpr int    N_LEVEL_2 = N_LEVEL * N_LEVEL;
static constexpr int    MAX_DEPTH = 2;
static constexpr int    N_HRCHY   = 3;
static constexpr int    MAIN_SIZE = N_LEVEL_2 * N_HRCHY; // 12

// ---------------------------------------------------------------------------
// RAII GPU buffer (mirrors DevBuf in test_linalg_engine_cuda.cu)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

using heom_liou_eigen = heom_liou<dynamic, c128, dense_matrix, row_major, row_major, eigen>;
using heom_hilb_eigen = heom_hilb<dynamic, c128, dense_matrix, row_major, eigen>;
using heom_ado_eigen  = heom_ado <dynamic, c128, dense_matrix, row_major, row_major, eigen>;
using heom_liou_cuda  = heom_liou<dynamic, c128, dense_matrix, row_major, row_major, cuda>;
using heom_hilb_cuda  = heom_hilb<dynamic, c128, dense_matrix, row_major, cuda>;
using heom_ado_cuda   = heom_ado <dynamic, c128, dense_matrix, row_major, row_major, cuda>;

// ---------------------------------------------------------------------------
// Common bath/Hamiltonian parameters applied to any heom_liou or heom_hilb.
// ---------------------------------------------------------------------------

template<typename H_t>
void setup_heom_params(H_t* h)
{
    h->H.set_shape(N_LEVEL, N_LEVEL);
    h->H.push(0, 0, {1.0, 0.0});
    h->n_level   = N_LEVEL;
    h->n_level_2 = N_LEVEL_2;

    h->alloc_noises(1);

    h->V[0].set_shape(N_LEVEL, N_LEVEL);
    h->V[0].push(0, 1, {1.0, 0.0});
    h->V[0].push(1, 0, {1.0, 0.0});

    h->len_gamma[0] = 1;
    h->gamma[0].set_shape(1, 1);
    h->gamma[0].push(0, 0, {-1.0, 0.0});

    h->phi_0[0]  = { c128{1.0, 0.0} };
    h->sigma[0]  = { c128{0.5, 0.0} };

    h->S[0].set_shape(1, 1);
    h->S[0].push(0, 0, {0.3, 0.0});
    h->A[0].set_shape(1, 1);
    h->A[0].push(0, 0, {0.1, 0.0});
    h->s_delta[0] = {0.0, 0.0};
}

// ---------------------------------------------------------------------------
// Non-trivial initial state used by all three tests.
// level 0: |0><0|            rho[0..3]  = {1,0,0,0}
// level 1: Pauli-X           rho[4..7]  = {0,1,1,0}
// level 2: fully mixed       rho[8..11] = {0.5,0.5,0.5,0.5}
// ---------------------------------------------------------------------------

static void fill_rho(std::vector<c128>& rho)
{
    rho.assign(MAIN_SIZE, {0.0, 0.0});
    rho[0]  = {1.0,  0.0};
    rho[5]  = {1.0,  0.0};
    rho[6]  = {1.0,  0.0};
    rho[8]  = {0.5,  0.0};
    rho[9]  = {0.5,  0.0};
    rho[10] = {0.5,  0.0};
    rho[11] = {0.5,  0.0};
}

// ---------------------------------------------------------------------------
// HeomDynamicsCuda.LiouvilleCUDAvsEigen
// ---------------------------------------------------------------------------

TEST(HeomDynamicsCuda, LiouvilleCUDAvsEigen)
{
    std::vector<c128> rho(MAIN_SIZE);
    fill_rho(rho);

    // --- Eigen reference ---
    eigen eng_e;
    auto* h_e = new heom_liou_eigen(MAX_DEPTH, 1, 1);
    setup_heom_params(h_e);
    h_e->set_param(&eng_e);

    int temp_sz_e = h_e->temp_size();
    std::vector<c128> temp_e(temp_sz_e, {0.0, 0.0});
    std::vector<c128> drho_dt_e(MAIN_SIZE, {0.0, 0.0});

    h_e->calc_diff_impl(&eng_e, drho_dt_e.data(), rho.data(),
                        c128{1.0, 0.0}, c128{0.0, 0.0}, temp_e.data());
    delete h_e;

    // --- CUDA ---
    // n_outer=1: OMP loop runs with 1 thread; child 0's stream handles GPU calls.
    cuda eng_c{0};
    auto* h_c = new heom_liou_cuda(MAX_DEPTH, 1, 1);
    setup_heom_params(h_c);
    h_c->set_param(&eng_c);

    int temp_sz_c = h_c->temp_size(); // N_LEVEL_2 * 3 * 1 = 12
    DevBuf<c128> d_rho(rho.data(), MAIN_SIZE);
    DevBuf<c128> d_drho_dt(MAIN_SIZE);
    DevBuf<c128> d_temp(temp_sz_c);

    h_c->calc_diff_impl(&eng_c, d_drho_dt.ptr, d_rho.ptr,
                        c128{1.0, 0.0}, c128{0.0, 0.0}, d_temp.ptr);

    std::vector<c128> drho_dt_c(MAIN_SIZE, {0.0, 0.0});
    d_drho_dt.to_host(drho_dt_c.data());
    cudaStreamSynchronize(eng_c.stream);
    delete h_c;

    for (int i = 0; i < MAIN_SIZE; ++i) {
        EXPECT_NEAR(drho_dt_c[i].real(), drho_dt_e[i].real(), EPS)
            << "real mismatch at index " << i;
        EXPECT_NEAR(drho_dt_c[i].imag(), drho_dt_e[i].imag(), EPS)
            << "imag mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// HeomDynamicsCuda.HilbertCUDAvsEigen
// ---------------------------------------------------------------------------

TEST(HeomDynamicsCuda, HilbertCUDAvsEigen)
{
    std::vector<c128> rho(MAIN_SIZE);
    fill_rho(rho);

    // --- Eigen reference ---
    eigen eng_e;
    auto* h_e = new heom_hilb_eigen(MAX_DEPTH, 1, 1);
    setup_heom_params(h_e);
    h_e->set_param(&eng_e);

    // temp_size = n_level * n_level * 3 * n_outer_threads = 4 * 3 * 1 = 12
    int temp_sz_e = h_e->temp_size();
    std::vector<c128> temp_e(temp_sz_e, {0.0, 0.0});
    std::vector<c128> drho_dt_e(MAIN_SIZE, {0.0, 0.0});

    h_e->calc_diff_impl(&eng_e, drho_dt_e.data(), rho.data(),
                        c128{1.0, 0.0}, c128{0.0, 0.0}, temp_e.data());
    delete h_e;

    // --- CUDA ---
    cuda eng_c{0};
    auto* h_c = new heom_hilb_cuda(MAX_DEPTH, 1, 1);
    setup_heom_params(h_c);
    h_c->set_param(&eng_c);

    int temp_sz_c = h_c->temp_size();
    DevBuf<c128> d_rho(rho.data(), MAIN_SIZE);
    DevBuf<c128> d_drho_dt(MAIN_SIZE);
    DevBuf<c128> d_temp(temp_sz_c);

    h_c->calc_diff_impl(&eng_c, d_drho_dt.ptr, d_rho.ptr,
                        c128{1.0, 0.0}, c128{0.0, 0.0}, d_temp.ptr);

    std::vector<c128> drho_dt_c(MAIN_SIZE, {0.0, 0.0});
    d_drho_dt.to_host(drho_dt_c.data());
    cudaStreamSynchronize(eng_c.stream);
    delete h_c;

    for (int i = 0; i < MAIN_SIZE; ++i) {
        EXPECT_NEAR(drho_dt_c[i].real(), drho_dt_e[i].real(), EPS)
            << "real mismatch at index " << i;
        EXPECT_NEAR(drho_dt_c[i].imag(), drho_dt_e[i].imag(), EPS)
            << "imag mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// HeomDynamicsCuda.AdoCUDAvsEigen
// ---------------------------------------------------------------------------

TEST(HeomDynamicsCuda, AdoCUDAvsEigen)
{
    std::vector<c128> rho(MAIN_SIZE);
    fill_rho(rho);

    // --- Eigen reference ---
    eigen eng_e;
    auto* h_e = new heom_ado_eigen(MAX_DEPTH, 1, 1);
    setup_heom_params(h_e);
    h_e->set_param(&eng_e);

    // heom_ado::temp_size() == 0; pass nullptr.
    std::vector<c128> drho_dt_e(MAIN_SIZE, {0.0, 0.0});

    h_e->calc_diff_impl(&eng_e, drho_dt_e.data(), rho.data(),
                        c128{1.0, 0.0}, c128{0.0, 0.0}, nullptr);
    delete h_e;

    // --- CUDA ---
    cuda eng_c{0};
    auto* h_c = new heom_ado_cuda(MAX_DEPTH, 1, 1);
    setup_heom_params(h_c);
    h_c->set_param(&eng_c);

    DevBuf<c128> d_rho(rho.data(), MAIN_SIZE);
    DevBuf<c128> d_drho_dt(MAIN_SIZE);

    // temp_size()==0; pass nullptr for temp_base.
    h_c->calc_diff_impl(&eng_c, d_drho_dt.ptr, d_rho.ptr,
                        c128{1.0, 0.0}, c128{0.0, 0.0}, nullptr);

    std::vector<c128> drho_dt_c(MAIN_SIZE, {0.0, 0.0});
    d_drho_dt.to_host(drho_dt_c.data());
    cudaStreamSynchronize(eng_c.stream);
    delete h_c;

    for (int i = 0; i < MAIN_SIZE; ++i) {
        EXPECT_NEAR(drho_dt_c[i].real(), drho_dt_e[i].real(), EPS)
            << "real mismatch at index " << i;
        EXPECT_NEAR(drho_dt_c[i].imag(), drho_dt_e[i].imag(), EPS)
            << "imag mismatch at index " << i;
    }
}

} // namespace
