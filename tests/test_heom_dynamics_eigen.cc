/* -*- mode:c++ -*-
 * LibHEOM -- GoogleTest: Eigen heom_hilb/heom_liou/heom_ado cross-space dynamics
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LICENSE.txt for licence.
 *
 * Tests that heom_hilb, heom_liou, and heom_ado produce identical drho_dt arrays
 * for the same input rho.  All three implement the same HEOM equations but via
 * different algorithms:
 *   heom_hilb:  gemm on n x n density matrices   (Hilbert space)
 *   heom_liou:  gemv on n^2-vectors              (Liouville space)
 *   heom_ado:   single global gemv on full R mat (ADO space)
 *
 * Physical system:
 *   n_level = 2, H = diag(1, 0), V = Pauli-X [[0,1],[1,0]]
 *   1 noise, 1 Matsubara pole: gamma = -1.0, phi_0 = 1.0, sigma = 0.5
 *   S = [[0.3]], A = [[0.1]], s_delta = 0.0
 *   truncation_depth = 2  ->  n_hierarchy = 3, main_size = 12
 *
 * Initial rho (non-trivial to exercise all hierarchy coupling terms):
 *   level 0: [[1, 0], [0, 0]]              -> rho[0..3]  = {1,0,0,0}
 *   level 1: [[0, 1], [1, 0]]              -> rho[4..7]  = {0,1,1,0}
 *   level 2: [[0.5,0.5],[0.5,0.5]]         -> rho[8..11] = {0.5,0.5,0.5,0.5}
 *------------------------------------------------------------------------*/

#include <gtest/gtest.h>
#include "libheom.h"
#include <vector>

namespace {
using namespace libheom;
using c128 = complex128;

using heom_hilb_t = heom_hilb<dynamic, c128, dense_matrix, row_major, eigen>;
using heom_liou_t = heom_liou<dynamic, c128, dense_matrix, row_major, row_major, eigen>;
using heom_ado_t  = heom_ado <dynamic, c128, dense_matrix, row_major, row_major, eigen>;

static constexpr int N_LEVEL   = 2;
static constexpr int N_LEVEL_2 = 4;
static constexpr int MAX_DEPTH = 2;
static constexpr int N_HRCHY  = 3;
static constexpr int MAIN_SIZE = N_LEVEL_2 * N_HRCHY; // 12

static constexpr double EPS_CROSS_SPACE = 1e-13;

// ---------------------------------------------------------------------------
// Fill common bath parameters (same for all space types)
// ---------------------------------------------------------------------------

template<typename QME>
static void fill_params(QME* h)
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
    h->phi_0[0] = { c128{1.0, 0.0} };
    h->sigma[0] = { c128{0.5, 0.0} };

    h->S[0].set_shape(1, 1);
    h->S[0].push(0, 0, {0.3, 0.0});
    h->A[0].set_shape(1, 1);
    h->A[0].push(0, 0, {0.1, 0.0});
    h->s_delta[0] = {0.0, 0.0};
}

// Build and initialise a solver object; caller owns the pointer.
static heom_hilb_t* make_hilb(eigen& eng, int n_outer=1, int n_inner=1)
{
    auto* h = new heom_hilb_t(MAX_DEPTH, n_inner, n_outer);
    fill_params(h);
    h->set_param(&eng);
    return h;
}

static heom_liou_t* make_liou(eigen& eng, int n_outer=1, int n_inner=1)
{
    auto* h = new heom_liou_t(MAX_DEPTH, n_inner, n_outer);
    fill_params(h);
    h->set_param(&eng);
    return h;
}

static heom_ado_t* make_ado(eigen& eng, int n_outer=1, int n_inner=1)
{
    auto* h = new heom_ado_t(MAX_DEPTH, n_inner, n_outer);
    fill_params(h);
    h->set_param(&eng);
    return h;
}

// Non-trivial initial rho: exercises +1/-1 hierarchy coupling
static std::vector<c128> make_rho()
{
    std::vector<c128> rho(MAIN_SIZE, {0.0, 0.0});
    // level 0: [[1,0],[0,0]]
    rho[0] = {1.0, 0.0};
    // level 1: [[0,1],[1,0]]
    rho[5] = {1.0, 0.0};
    rho[6] = {1.0, 0.0};
    // level 2: [[0.5,0.5],[0.5,0.5]]
    rho[8]  = {0.5, 0.0};
    rho[9]  = {0.5, 0.0};
    rho[10] = {0.5, 0.0};
    rho[11] = {0.5, 0.0};
    return rho;
}

// Run calc_diff_impl and return drho_dt
template<typename Solver>
static std::vector<c128> run_diff(Solver* h, eigen& eng, const std::vector<c128>& rho)
{
    int temp_sz = h->temp_size();
    std::vector<c128> temp(temp_sz, {0.0, 0.0});
    std::vector<c128> drho_dt(MAIN_SIZE, {0.0, 0.0});
    h->calc_diff_impl(&eng, drho_dt.data(),
                      const_cast<c128*>(rho.data()),
                      c128{1.0, 0.0}, c128{0.0, 0.0},
                      temp.data());
    return drho_dt;
}

// ---------------------------------------------------------------------------
// Sanity: zero rho gives zero drho_dt for all space types
// ---------------------------------------------------------------------------

TEST(HeomDynamicsEigen, ZeroRhoGivesZero_Hilbert) {
    eigen eng;
    auto* h = make_hilb(eng);
    std::vector<c128> rho(MAIN_SIZE, {0.0, 0.0});
    auto drho = run_diff(h, eng, rho);
    for (int i = 0; i < MAIN_SIZE; ++i) {
        EXPECT_NEAR(drho[i].real(), 0.0, 1e-15) << "index " << i;
        EXPECT_NEAR(drho[i].imag(), 0.0, 1e-15) << "index " << i;
    }
    delete h;
}

TEST(HeomDynamicsEigen, ZeroRhoGivesZero_Liouville) {
    eigen eng;
    auto* h = make_liou(eng);
    std::vector<c128> rho(MAIN_SIZE, {0.0, 0.0});
    auto drho = run_diff(h, eng, rho);
    for (int i = 0; i < MAIN_SIZE; ++i) {
        EXPECT_NEAR(drho[i].real(), 0.0, 1e-15) << "index " << i;
        EXPECT_NEAR(drho[i].imag(), 0.0, 1e-15) << "index " << i;
    }
    delete h;
}

TEST(HeomDynamicsEigen, ZeroRhoGivesZero_Ado) {
    eigen eng;
    auto* h = make_ado(eng);
    std::vector<c128> rho(MAIN_SIZE, {0.0, 0.0});
    auto drho = run_diff(h, eng, rho);
    for (int i = 0; i < MAIN_SIZE; ++i) {
        EXPECT_NEAR(drho[i].real(), 0.0, 1e-15) << "index " << i;
        EXPECT_NEAR(drho[i].imag(), 0.0, 1e-15) << "index " << i;
    }
    delete h;
}

// ---------------------------------------------------------------------------
// Cross-space consistency: hilb == liou == ado for the same non-trivial rho.
//
// All three spaces encode the same HEOM equations; the drho_dt array (row-major
// vectorized density matrices per hierarchy node) must be element-for-element
// identical across representations.
// ---------------------------------------------------------------------------

TEST(HeomDynamicsEigen, CrossSpace_HilbertMatchesLiouville) {
    eigen eng;
    auto rho = make_rho();

    auto* hilb = make_hilb(eng);
    auto* liou = make_liou(eng);

    auto drho_hilb = run_diff(hilb, eng, rho);
    auto drho_liou = run_diff(liou, eng, rho);

    for (int i = 0; i < MAIN_SIZE; ++i) {
        EXPECT_NEAR(drho_hilb[i].real(), drho_liou[i].real(), EPS_CROSS_SPACE)
            << "real mismatch at index " << i;
        EXPECT_NEAR(drho_hilb[i].imag(), drho_liou[i].imag(), EPS_CROSS_SPACE)
            << "imag mismatch at index " << i;
    }
    delete hilb;
    delete liou;
}

TEST(HeomDynamicsEigen, CrossSpace_LiouvilleMatchesAdo) {
    eigen eng;
    auto rho = make_rho();

    auto* liou = make_liou(eng);
    auto* ado  = make_ado(eng);

    auto drho_liou = run_diff(liou, eng, rho);
    auto drho_ado  = run_diff(ado,  eng, rho);

    for (int i = 0; i < MAIN_SIZE; ++i) {
        EXPECT_NEAR(drho_liou[i].real(), drho_ado[i].real(), EPS_CROSS_SPACE)
            << "real mismatch at index " << i;
        EXPECT_NEAR(drho_liou[i].imag(), drho_ado[i].imag(), EPS_CROSS_SPACE)
            << "imag mismatch at index " << i;
    }
    delete liou;
    delete ado;
}

TEST(HeomDynamicsEigen, CrossSpace_HilbertMatchesAdo) {
    eigen eng;
    auto rho = make_rho();

    auto* hilb = make_hilb(eng);
    auto* ado  = make_ado(eng);

    auto drho_hilb = run_diff(hilb, eng, rho);
    auto drho_ado  = run_diff(ado,  eng, rho);

    for (int i = 0; i < MAIN_SIZE; ++i) {
        EXPECT_NEAR(drho_hilb[i].real(), drho_ado[i].real(), EPS_CROSS_SPACE)
            << "real mismatch at index " << i;
        EXPECT_NEAR(drho_hilb[i].imag(), drho_ado[i].imag(), EPS_CROSS_SPACE)
            << "imag mismatch at index " << i;
    }
    delete hilb;
    delete ado;
}

// ---------------------------------------------------------------------------
// n_outer_threads > 1 gives identical result to n_outer=1 for all spaces
// ---------------------------------------------------------------------------

TEST(HeomDynamicsEigen, NOuterThreads_Hilbert) {
    eigen eng;
    auto rho = make_rho();

    auto* h1 = make_hilb(eng, 1);
    auto* h4 = make_hilb(eng, 4);

    auto drho_1 = run_diff(h1, eng, rho);
    auto drho_4 = run_diff(h4, eng, rho);

    for (int i = 0; i < MAIN_SIZE; ++i) {
        EXPECT_EQ(drho_1[i].real(), drho_4[i].real()) << "index " << i;
        EXPECT_EQ(drho_1[i].imag(), drho_4[i].imag()) << "index " << i;
    }
    delete h1;
    delete h4;
}

TEST(HeomDynamicsEigen, NOuterThreads_Liouville) {
    eigen eng;
    auto rho = make_rho();

    auto* h1 = make_liou(eng, 1);
    auto* h4 = make_liou(eng, 4);

    auto drho_1 = run_diff(h1, eng, rho);
    auto drho_4 = run_diff(h4, eng, rho);

    for (int i = 0; i < MAIN_SIZE; ++i) {
        EXPECT_EQ(drho_1[i].real(), drho_4[i].real()) << "index " << i;
        EXPECT_EQ(drho_1[i].imag(), drho_4[i].imag()) << "index " << i;
    }
    delete h1;
    delete h4;
}

} // namespace
