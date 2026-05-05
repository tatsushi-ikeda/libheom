/* -*- mode:c++ -*-
 * LibHEOM -- unit tests for hierarchy_space
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LICENSE.txt for licence.
 *------------------------------------------------------------------------*/

#include <gtest/gtest.h>
#include <numeric>
#include "hierarchy_space.h"

namespace {
using namespace libheom;

// Stars-and-bars: C(K+N, N) = C(K+N, K)
static int multicomb(int K, int N) {
    // number of ADOs = number of K-tuples with sum <= N
    long long num = 1, den = 1;
    for (int i = 0; i < N; ++i) {
        num *= K + i + 1;
        den *= i + 1;
        long long g = std::__gcd(num, den);
        num /= g; den /= g;
    }
    return static_cast<int>(num / den);
}

static hierarchy_space build(int K, int N) {
    hierarchy_space hs;
    hs.n_modes = K;
    alloc_hierarchy_space(hs, N);
    return hs;
}

// ---------------------------------------------------------------------------
// n_hierarchy = C(K+N, N)
// ---------------------------------------------------------------------------

TEST(HrchySpace, Count_K1) {
    for (int N = 0; N <= 5; ++N) {
        auto hs = build(1, N);
        EXPECT_EQ(static_cast<int>(hs.n.size()), multicomb(1, N))
            << "K=1 N=" << N;
    }
}

TEST(HrchySpace, Count_K2) {
    for (int N = 0; N <= 4; ++N) {
        auto hs = build(2, N);
        EXPECT_EQ(static_cast<int>(hs.n.size()), multicomb(2, N))
            << "K=2 N=" << N;
    }
}

TEST(HrchySpace, Count_K3) {
    for (int N = 0; N <= 3; ++N) {
        auto hs = build(3, N);
        EXPECT_EQ(static_cast<int>(hs.n.size()), multicomb(3, N))
            << "K=3 N=" << N;
    }
}

TEST(HrchySpace, Count_K4) {
    for (int N = 1; N <= 3; ++N) {
        auto hs = build(4, N);
        EXPECT_EQ(static_cast<int>(hs.n.size()), multicomb(4, N))
            << "K=4 N=" << N;
    }
}

// ---------------------------------------------------------------------------
// ptr_void == n_hierarchy (the extra slot beyond valid ADOs)
// ---------------------------------------------------------------------------

TEST(HrchySpace, PtrVoidEqualsNHrchy) {
    auto hs = build(2, 3);
    int n_hierarchy = static_cast<int>(hs.n.size());
    EXPECT_EQ(hs.ptr_void, n_hierarchy);
}

// ---------------------------------------------------------------------------
// ptr_p1 o ptr_m1 = identity
//   if ptr_p1[i][k] != ptr_void -> ptr_m1[ptr_p1[i][k]][k] == i
//   if ptr_m1[i][k] != ptr_void -> ptr_p1[ptr_m1[i][k]][k] == i
// ---------------------------------------------------------------------------

TEST(HrchySpace, PtrInverseK1N3) {
    auto hs = build(1, 3);
    int n_hierarchy = static_cast<int>(hs.n.size());
    int K = hs.n_modes;
    for (int i = 0; i < n_hierarchy; ++i) {
        for (int k = 0; k < K; ++k) {
            int ip1 = hs.ptr_p1[i][k];
            if (ip1 != hs.ptr_void) {
                EXPECT_EQ(hs.ptr_m1[ip1][k], i) << "i=" << i << " k=" << k;
            }
            int im1 = hs.ptr_m1[i][k];
            if (im1 != hs.ptr_void) {
                EXPECT_EQ(hs.ptr_p1[im1][k], i) << "i=" << i << " k=" << k;
            }
        }
    }
}

TEST(HrchySpace, PtrInverseK3N2) {
    auto hs = build(3, 2);
    int n_hierarchy = static_cast<int>(hs.n.size());
    int K = hs.n_modes;
    for (int i = 0; i < n_hierarchy; ++i) {
        for (int k = 0; k < K; ++k) {
            int ip1 = hs.ptr_p1[i][k];
            if (ip1 != hs.ptr_void) {
                EXPECT_EQ(hs.ptr_m1[ip1][k], i) << "i=" << i << " k=" << k;
            }
            int im1 = hs.ptr_m1[i][k];
            if (im1 != hs.ptr_void) {
                EXPECT_EQ(hs.ptr_p1[im1][k], i) << "i=" << i << " k=" << k;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Zero-tier ADO (index 0) always exists and has all-zero index vector
// ---------------------------------------------------------------------------

TEST(HrchySpace, ZeroTierIsFirst) {
    auto hs = build(2, 3);
    EXPECT_EQ(static_cast<int>(hs.n.size()) > 0, true);
    // First ADO should be the all-zeros index
    for (int k = 0; k < hs.n_modes; ++k) {
        EXPECT_EQ(hs.n[0][k], 0);
    }
}

// ---------------------------------------------------------------------------
// ptr_void slot: its own p1/m1 are all ptr_void (sentinel loops on itself)
// ---------------------------------------------------------------------------

TEST(HrchySpace, PtrVoidSelfLoops) {
    auto hs = build(2, 2);
    int pv = hs.ptr_void;
    for (int k = 0; k < hs.n_modes; ++k) {
        EXPECT_EQ(hs.ptr_p1[pv][k], pv);
        EXPECT_EQ(hs.ptr_m1[pv][k], pv);
    }
}

} // namespace
