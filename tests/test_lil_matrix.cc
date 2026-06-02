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

// ---------------------------------------------------------------------------
// push / exists
// ---------------------------------------------------------------------------

TEST(LilMatrix, PushCreatesEntry) {
    lil_matrix<dynamic, c128, row_major, nil> m;
    m.set_shape(3, 3);
    m.push(0, 1, {1.0, 0.0});
    m.push(2, 0, {0.0, 2.5});
    EXPECT_TRUE(m.exists(0, 1));
    EXPECT_TRUE(m.exists(2, 0));
    EXPECT_FALSE(m.exists(0, 0));
    EXPECT_FALSE(m.exists(1, 2));
}

TEST(LilMatrix, PushAccumulates) {
    // push() uses +=, not assignment
    lil_matrix<dynamic, c128, row_major, nil> m;
    m.set_shape(2, 2);
    m.push(0, 0, {1.0, 0.0});
    m.push(0, 0, {2.0, 1.0});
    EXPECT_EQ(m.data.at(0).at(0), c128(3.0, 1.0));
}

// ---------------------------------------------------------------------------
// optimize: removes entries whose magnitude <= tol * max_magnitude
// ---------------------------------------------------------------------------

TEST(LilMatrix, OptimizeRemovesNearZero) {
    lil_matrix<dynamic, c128, row_major, nil> m;
    m.set_shape(3, 3);
    m.push(0, 0, {1.0, 0.0});   // max entry
    m.push(1, 1, {1e-16, 0.0}); // |val| <= eps * max -> removed
    EXPECT_TRUE(m.exists(0, 0));
    EXPECT_TRUE(m.exists(1, 1));
    m.optimize();
    EXPECT_TRUE(m.exists(0, 0));
    EXPECT_FALSE(m.exists(1, 1));
}

TEST(LilMatrix, OptimizeKeepsSignificantEntries) {
    lil_matrix<dynamic, c128, row_major, nil> m;
    m.set_shape(2, 2);
    m.push(0, 0, {1.0, 0.0});
    m.push(0, 1, {0.5, 0.0}); // 0.5 * max == 0.5 > eps
    m.optimize();
    EXPECT_TRUE(m.exists(0, 0));
    EXPECT_TRUE(m.exists(0, 1));
}

// ---------------------------------------------------------------------------
// set_identity
// ---------------------------------------------------------------------------

TEST(LilMatrix, SetIdentity) {
    lil_matrix<dynamic, c128, row_major, nil> m;
    m.set_identity(3);
    EXPECT_EQ(std::get<0>(m.shape), 3);
    EXPECT_EQ(std::get<1>(m.shape), 3);
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(m.exists(i, i));
        EXPECT_EQ(m.data.at(i).at(i), c128(1.0, 0.0));
        for (int j = 0; j < 3; ++j) {
            if (i != j) EXPECT_FALSE(m.exists(i, j));
        }
    }
}

// ---------------------------------------------------------------------------
// set_adjoint: conjugate-transpose
// ---------------------------------------------------------------------------

TEST(LilMatrix, SetAdjoint) {
    lil_matrix<dynamic, c128, row_major, nil> src, adj;
    src.set_shape(2, 3);
    src.push(0, 1, {1.0,  2.0});
    src.push(1, 2, {3.0, -1.0});
    adj.set_adjoint(src);
    // adj[j][i] == conj(src[i][j])
    EXPECT_EQ(adj.data.at(1).at(0), c128(1.0, -2.0));
    EXPECT_EQ(adj.data.at(2).at(1), c128(3.0,  1.0));
    EXPECT_EQ(std::get<0>(adj.shape), 3);
    EXPECT_EQ(std::get<1>(adj.shape), 2);
}

// ---------------------------------------------------------------------------
// clear
// ---------------------------------------------------------------------------

TEST(LilMatrix, ClearRemovesAllEntries) {
    lil_matrix<dynamic, c128, row_major, nil> m;
    m.set_shape(2, 2);
    m.push(0, 0, {1.0, 0.0});
    m.push(1, 1, {2.0, 0.0});
    m.clear();
    EXPECT_TRUE(m.data.empty());
}

} // namespace
