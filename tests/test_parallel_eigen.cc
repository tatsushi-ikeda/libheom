/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LICENSE.txt for licence.
 *------------------------------------------------------------------------*/

// System: 2-level, H = diag(1,0), V = Pauli-X
//   Bath: 1 noise, 1 Matsubara pole (overdamped Drude approximation)
//         gamma = -1.0, phi_0 = 1.0, sigma = 0.5
//         S = [[0.3]], A = [[0.1]], s_delta = 0.0
//   truncation_depth = 2 -> n_hierarchy = 3 (hierarchy states: {0}, {1}, {2})

#include <gtest/gtest.h>
#include "libheom.h"
#include <vector>
#include <cstring>

namespace {
using namespace libheom;
using c128 = complex128;

// heom_liou<dynamic, c128, dense_matrix, row_major, row_major, eigen>
using heom_liou_t = heom_liou<dynamic, c128, dense_matrix, row_major, row_major, eigen>;

static constexpr int N_LEVEL   = 2;
static constexpr int N_LEVEL_2 = N_LEVEL * N_LEVEL;
static constexpr int MAX_DEPTH = 2;
static constexpr int N_HRCHY  = 3; // C(MAX_DEPTH + 1, 1) = 3 for 1D hierarchy
static constexpr int MAIN_SIZE = N_LEVEL_2 * N_HRCHY; // 12

// ---------------------------------------------------------------------------
// Build a minimal heom_liou with the given n_outer_threads.
// Call set_param() so the internal Liouvillian matrices are initialised.
// ---------------------------------------------------------------------------

static heom_liou_t* make_heom(eigen& eng, int n_outer, int n_inner = 1)
{
    auto* h = new heom_liou_t(MAX_DEPTH, n_inner, n_outer);

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

    h->s_mat[0].set_shape(1, 1);
    h->s_mat[0].push(0, 0, {0.3, 0.0});
    h->a_mat[0].set_shape(1, 1);
    h->a_mat[0].push(0, 0, {0.1, 0.0});
    h->s_delta[0] = {0.0, 0.0};

    h->set_param(&eng);
    return h;
}

// ---------------------------------------------------------------------------
// n_outer_threads: calc_time_derivative with 1 vs 4 threads must be bit-identical.
//
// The OMP parallel loop assigns each hierarchy node (lidx) to a thread.
// Each thread writes only to drho_dt[lidx*n_level_2 .. (lidx+1)*n_level_2-1]
// via a per-thread temp buffer, so there are no shared writes.  The
// floating-point operations for a given lidx are identical regardless of how
// many other threads are running concurrently.
// ---------------------------------------------------------------------------

TEST(ParallelEigen, NOuterThreadsGiveIdenticalDerivative) {
    eigen eng;

    auto* h1 = make_heom(eng, 1);
    auto* h4 = make_heom(eng, 4);

    ASSERT_EQ(h1->n_hierarchy, N_HRCHY);
    ASSERT_EQ(h4->n_hierarchy, N_HRCHY);
    ASSERT_EQ(h1->main_size(), MAIN_SIZE);

    // Non-trivial initial state: rho at hierarchy level 0 = |0><0|
    std::vector<c128> rho(MAIN_SIZE, {0.0, 0.0});
    rho[0] = {1.0, 0.0};

    int temp_sz_1 = h1->temp_size(); // N_LEVEL_2 * 3 * 1 = 12
    int temp_sz_4 = h4->temp_size(); // N_LEVEL_2 * 3 * 4 = 48
    std::vector<c128> temp_1(temp_sz_1, {0.0, 0.0});
    std::vector<c128> temp_4(temp_sz_4, {0.0, 0.0});
    std::vector<c128> drho_dt_1(MAIN_SIZE, {0.0, 0.0});
    std::vector<c128> drho_dt_4(MAIN_SIZE, {0.0, 0.0});

    h1->calc_time_derivative(&eng, drho_dt_1.data(), rho.data(),
                             c128{1.0, 0.0}, c128{0.0, 0.0}, temp_1.data());

    h4->calc_time_derivative(&eng, drho_dt_4.data(), rho.data(),
                             c128{1.0, 0.0}, c128{0.0, 0.0}, temp_4.data());

    for (int i = 0; i < MAIN_SIZE; ++i) {
        EXPECT_EQ(drho_dt_1[i].real(), drho_dt_4[i].real())
            << "real part mismatch at index " << i;
        EXPECT_EQ(drho_dt_1[i].imag(), drho_dt_4[i].imag())
            << "imag part mismatch at index " << i;
    }

    delete h1;
    delete h4;
}

// ---------------------------------------------------------------------------
// n_inner_threads: calc_time_derivative with Eigen::setNbThreads(1) vs (4).
//
// Inner thread count controls Eigen gemv/gemm parallelism.  For the small
// 4x4 matrices in this test Eigen will likely use 1 thread regardless, but
// the API path is exercised.  Results are checked with a tight tolerance
// (1e-14) rather than exact equality because inner-thread reordering may
// introduce sub-ULP differences on some platforms.
// ---------------------------------------------------------------------------

TEST(ParallelEigen, NInnerThreadsGiveConsistentDerivative) {
    static constexpr double EPS = 1e-14;
    eigen eng;

    auto* h1 = make_heom(eng, 1, 1);
    auto* h4 = make_heom(eng, 1, 4);

    std::vector<c128> rho(MAIN_SIZE, {0.0, 0.0});
    rho[0] = {1.0, 0.0};

    int temp_sz = h1->temp_size();
    std::vector<c128> temp(temp_sz, {0.0, 0.0});
    std::vector<c128> drho_dt_1(MAIN_SIZE, {0.0, 0.0});
    std::vector<c128> drho_dt_4(MAIN_SIZE, {0.0, 0.0});

    h1->calc_time_derivative(&eng, drho_dt_1.data(), rho.data(),
                             c128{1.0, 0.0}, c128{0.0, 0.0}, temp.data());

    std::fill(temp.begin(), temp.end(), c128{0.0, 0.0});
    h4->calc_time_derivative(&eng, drho_dt_4.data(), rho.data(),
                             c128{1.0, 0.0}, c128{0.0, 0.0}, temp.data());

    for (int i = 0; i < MAIN_SIZE; ++i) {
        EXPECT_NEAR(drho_dt_1[i].real(), drho_dt_4[i].real(), EPS)
            << "real part mismatch at index " << i;
        EXPECT_NEAR(drho_dt_1[i].imag(), drho_dt_4[i].imag(), EPS)
            << "imag part mismatch at index " << i;
    }

    delete h1;
    delete h4;
}

// ---------------------------------------------------------------------------
// Sanity check: single-thread calc_time_derivative with all-zero rho gives zero.
// ---------------------------------------------------------------------------

TEST(ParallelEigen, ZeroRhoGivesZeroDerivative) {
    eigen eng;
    auto* h = make_heom(eng, 1);

    std::vector<c128> rho(MAIN_SIZE, {0.0, 0.0});
    int temp_sz = h->temp_size();
    std::vector<c128> temp(temp_sz, {0.0, 0.0});
    std::vector<c128> drho_dt(MAIN_SIZE, {99.0, 0.0}); // pre-fill to detect errors

    h->calc_time_derivative(&eng, drho_dt.data(), rho.data(),
                            c128{1.0, 0.0}, c128{0.0, 0.0}, temp.data());

    for (int i = 0; i < MAIN_SIZE; ++i) {
        EXPECT_NEAR(drho_dt[i].real(), 0.0, 1e-15) << "index " << i;
        EXPECT_NEAR(drho_dt[i].imag(), 0.0, 1e-15) << "index " << i;
    }

    delete h;
}

} // namespace
