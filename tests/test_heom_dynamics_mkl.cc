/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LICENSE.txt for licence.
 *------------------------------------------------------------------------*/

// System: n_level=2, H=diag(1,0), V=Pauli-X
//   Bath: 1 noise, 1 Matsubara pole
//         gamma=-1.0, phi_0=1.0, sigma=0.5
//         S=[[0.3]], A=[[0.1]], s_delta=0.0
//   truncation_depth=2 -> n_hierarchy=3, main_size=12

#include <gtest/gtest.h>
#include "libheom.h"
#include <vector>

namespace {
using namespace libheom;
using c128 = complex128;

static constexpr double EPS       = 1e-12;
static constexpr int    N_LEVEL   = 2;
static constexpr int    N_LEVEL_2 = N_LEVEL * N_LEVEL;
static constexpr int    MAX_DEPTH = 2;
static constexpr int    N_HRCHY   = 3;
static constexpr int    MAIN_SIZE = N_LEVEL_2 * N_HRCHY; // 12

using heom_liou_eigen = heom_liou<dynamic, c128, dense_matrix, row_major, row_major, eigen>;
using heom_hilb_eigen = heom_hilb<dynamic, c128, dense_matrix, row_major, eigen>;
using heom_ado_eigen  = heom_ado <dynamic, c128, dense_matrix, row_major, row_major, eigen>;
using heom_liou_mkl   = heom_liou<dynamic, c128, dense_matrix, row_major, row_major, mkl>;
using heom_hilb_mkl   = heom_hilb<dynamic, c128, dense_matrix, row_major, mkl>;
using heom_ado_mkl    = heom_ado <dynamic, c128, dense_matrix, row_major, row_major, mkl>;

// ---------------------------------------------------------------------------
// Common bath/Hamiltonian parameters applied to any solver type.
// ---------------------------------------------------------------------------

template<typename H_t>
static void setup_heom_params(H_t* h)
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

    h->s_mat[0].set_shape(1, 1);
    h->s_mat[0].push(0, 0, {0.3, 0.0});
    h->a_mat[0].set_shape(1, 1);
    h->a_mat[0].push(0, 0, {0.1, 0.0});
    h->s_delta[0] = {0.0, 0.0};
}

// ---------------------------------------------------------------------------
// Non-trivial initial state: exercises all hierarchy coupling terms.
// level 0: |0><0|      rho[0..3]  = {1,0,0,0}
// level 1: Pauli-X     rho[4..7]  = {0,1,1,0}
// level 2: fully mixed rho[8..11] = {0.5,0.5,0.5,0.5}
// ---------------------------------------------------------------------------

static std::vector<c128> make_rho()
{
    std::vector<c128> rho(MAIN_SIZE, {0.0, 0.0});
    rho[0]  = {1.0, 0.0};
    rho[5]  = {1.0, 0.0};
    rho[6]  = {1.0, 0.0};
    rho[8]  = {0.5, 0.0};
    rho[9]  = {0.5, 0.0};
    rho[10] = {0.5, 0.0};
    rho[11] = {0.5, 0.0};
    return rho;
}

// Run calc_time_derivative and return drho_dt for a host-side solver.
template<typename Solver, typename Eng>
static std::vector<c128> run_diff(Solver* h, Eng& eng,
                                  const std::vector<c128>& rho,
                                  c128* temp_buf)
{
    std::vector<c128> drho_dt(MAIN_SIZE, {0.0, 0.0});
    h->calc_time_derivative(&eng, drho_dt.data(),
                            const_cast<c128*>(rho.data()),
                      c128{1.0, 0.0}, c128{0.0, 0.0},
                      temp_buf);
    return drho_dt;
}

// ---------------------------------------------------------------------------
// LiouvilleMKLvsEigen
// ---------------------------------------------------------------------------

TEST(HeomDynamicsMkl, LiouvilleMKLvsEigen)
{
    auto rho = make_rho();

    eigen eng_e;
    auto* h_e = new heom_liou_eigen(MAX_DEPTH, 1, 1);
    setup_heom_params(h_e);
    h_e->set_param(&eng_e);
    int tsz_e = h_e->temp_size();
    std::vector<c128> temp_e(tsz_e, {0.0, 0.0});
    auto drho_e = run_diff(h_e, eng_e, rho, temp_e.data());
    delete h_e;

    mkl eng_m;
    auto* h_m = new heom_liou_mkl(MAX_DEPTH, 1, 1);
    setup_heom_params(h_m);
    h_m->set_param(&eng_m);
    int tsz_m = h_m->temp_size();
    std::vector<c128> temp_m(tsz_m, {0.0, 0.0});
    auto drho_m = run_diff(h_m, eng_m, rho, temp_m.data());
    delete h_m;

    for (int i = 0; i < MAIN_SIZE; ++i) {
        EXPECT_NEAR(drho_m[i].real(), drho_e[i].real(), EPS)
            << "real mismatch at index " << i;
        EXPECT_NEAR(drho_m[i].imag(), drho_e[i].imag(), EPS)
            << "imag mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// HilbertMKLvsEigen
// ---------------------------------------------------------------------------

TEST(HeomDynamicsMkl, HilbertMKLvsEigen)
{
    auto rho = make_rho();

    eigen eng_e;
    auto* h_e = new heom_hilb_eigen(MAX_DEPTH, 1, 1);
    setup_heom_params(h_e);
    h_e->set_param(&eng_e);
    int tsz_e = h_e->temp_size();
    std::vector<c128> temp_e(tsz_e, {0.0, 0.0});
    auto drho_e = run_diff(h_e, eng_e, rho, temp_e.data());
    delete h_e;

    mkl eng_m;
    auto* h_m = new heom_hilb_mkl(MAX_DEPTH, 1, 1);
    setup_heom_params(h_m);
    h_m->set_param(&eng_m);
    int tsz_m = h_m->temp_size();
    std::vector<c128> temp_m(tsz_m, {0.0, 0.0});
    auto drho_m = run_diff(h_m, eng_m, rho, temp_m.data());
    delete h_m;

    for (int i = 0; i < MAIN_SIZE; ++i) {
        EXPECT_NEAR(drho_m[i].real(), drho_e[i].real(), EPS)
            << "real mismatch at index " << i;
        EXPECT_NEAR(drho_m[i].imag(), drho_e[i].imag(), EPS)
            << "imag mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// AdoMKLvsEigen
// ---------------------------------------------------------------------------

TEST(HeomDynamicsMkl, AdoMKLvsEigen)
{
    auto rho = make_rho();

    eigen eng_e;
    auto* h_e = new heom_ado_eigen(MAX_DEPTH, 1, 1);
    setup_heom_params(h_e);
    h_e->set_param(&eng_e);
    // heom_ado::temp_size() == 0; pass nullptr.
    auto drho_e = run_diff(h_e, eng_e, rho, nullptr);
    delete h_e;

    mkl eng_m;
    auto* h_m = new heom_ado_mkl(MAX_DEPTH, 1, 1);
    setup_heom_params(h_m);
    h_m->set_param(&eng_m);
    auto drho_m = run_diff(h_m, eng_m, rho, nullptr);
    delete h_m;

    for (int i = 0; i < MAIN_SIZE; ++i) {
        EXPECT_NEAR(drho_m[i].real(), drho_e[i].real(), EPS)
            << "real mismatch at index " << i;
        EXPECT_NEAR(drho_m[i].imag(), drho_e[i].imag(), EPS)
            << "imag mismatch at index " << i;
    }
}

} // namespace
