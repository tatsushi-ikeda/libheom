/* -*- mode:c++ -*-
 * LibHEOM -- unit tests for heom_liou superoperator construction:
 *   Phi, Psi, Xi, R_0 symmetries
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LICENSE.txt for licence.
 *------------------------------------------------------------------------*/

#include <gtest/gtest.h>
#include "libheom.h"

namespace {
using namespace libheom;
using c128 = complex128;

// mkl_spblas.h may inject a global struct sparse_matrix; disambiguate.
using libheom::sparse_matrix;

static constexpr double EPS = 1e-14;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build Phi = i*(V(x)I - I(x)V^T) in Liouville space (row-major vec).
static lil_matrix<dynamic, c128, row_major, nil>
make_Phi(const lil_matrix<dynamic, c128, row_major, nil>& V)
{
    int n = std::get<0>(V.shape);
    lil_matrix<dynamic, c128, row_major, nil> Phi;
    Phi.set_shape(n*n, n*n);
    kron_x_1  <dynamic>(nilobj, +i_unit<c128>(), V, zero<c128>(), Phi);
    kron_1_x_T<dynamic>(nilobj, -i_unit<c128>(), V,  one<c128>(), Phi);
    Phi.optimize();
    return Phi;
}

// Build Psi = V(x)I + I(x)V^T
static lil_matrix<dynamic, c128, row_major, nil>
make_Psi(const lil_matrix<dynamic, c128, row_major, nil>& V)
{
    int n = std::get<0>(V.shape);
    lil_matrix<dynamic, c128, row_major, nil> Psi;
    Psi.set_shape(n*n, n*n);
    kron_x_1  <dynamic>(nilobj, +one<c128>(), V, zero<c128>(), Psi);
    kron_1_x_T<dynamic>(nilobj, +one<c128>(), V,  one<c128>(), Psi);
    Psi.optimize();
    return Psi;
}

// Build L = i*(H(x)I - I(x)H^T)  (free-system Liouvillian)
static lil_matrix<dynamic, c128, row_major, nil>
make_L(const lil_matrix<dynamic, c128, row_major, nil>& H)
{
    int n = std::get<0>(H.shape);
    lil_matrix<dynamic, c128, row_major, nil> L;
    L.set_shape(n*n, n*n);
    kron_x_1  <dynamic>(nilobj, +i_unit<c128>(), H, zero<c128>(), L);
    kron_1_x_T<dynamic>(nilobj, -i_unit<c128>(), H,  one<c128>(), L);
    L.optimize();
    return L;
}

// Return the (row, col) element; 0 if not present.
static c128 entry(const lil_matrix<dynamic, c128, row_major, nil>& m, int row, int col)
{
    auto ri = m.data.find(row);
    if (ri == m.data.end()) return c128{0, 0};
    auto ci = ri->second.find(col);
    if (ci == ri->second.end()) return c128{0, 0};
    return ci->second;
}

// ---------------------------------------------------------------------------
// 2x2 operators used in multiple tests
//   V = Pauli-Z = diag(1, -1)   (Hermitian, real)
//   H = Pauli-X = [[0,1],[1,0]] (Hermitian, real)
// ---------------------------------------------------------------------------

static lil_matrix<dynamic, c128, row_major, nil> make_pauli_z()
{
    lil_matrix<dynamic, c128, row_major, nil> V;
    V.set_shape(2, 2);
    V.push(0, 0, {+1.0, 0.0});
    V.push(1, 1, {-1.0, 0.0});
    return V;
}

static lil_matrix<dynamic, c128, row_major, nil> make_pauli_x()
{
    lil_matrix<dynamic, c128, row_major, nil> H;
    H.set_shape(2, 2);
    H.push(0, 1, {1.0, 0.0});
    H.push(1, 0, {1.0, 0.0});
    return H;
}

// ---------------------------------------------------------------------------
// Phi = i*[V, .]  is anti-Hermitian: Phi^dag = -Phi
// For Pauli-Z: Phi has non-zeros only at (1,1)=2i and (2,2)=-2i.
// ---------------------------------------------------------------------------

TEST(HeomConstruction, Phi_AntiHermitian) {
    auto V   = make_pauli_z();
    auto Phi = make_Phi(V);

    // Expected non-zeros: Phi[1][1] = 2i, Phi[2][2] = -2i
    EXPECT_NEAR(entry(Phi, 1, 1).real(), 0.0, EPS);
    EXPECT_NEAR(entry(Phi, 1, 1).imag(), 2.0, EPS);
    EXPECT_NEAR(entry(Phi, 2, 2).real(), 0.0, EPS);
    EXPECT_NEAR(entry(Phi, 2, 2).imag(),-2.0, EPS);

    // All off-diagonal terms are zero for diagonal V
    EXPECT_NEAR(std::abs(entry(Phi, 0, 0)), 0.0, EPS);
    EXPECT_NEAR(std::abs(entry(Phi, 3, 3)), 0.0, EPS);

    // Check Phi^dag = -Phi via set_adjoint
    lil_matrix<dynamic, c128, row_major, nil> Phi_adj;
    Phi_adj.set_adjoint(Phi);

    // Phi^dag[1][1] should equal -Phi[1][1] = -2i
    EXPECT_NEAR(entry(Phi_adj, 1, 1).imag(), -2.0, EPS);
    // Phi^dag[2][2] should equal -Phi[2][2] = +2i
    EXPECT_NEAR(entry(Phi_adj, 2, 2).imag(), +2.0, EPS);
}

// ---------------------------------------------------------------------------
// Psi = {V, .}  is Hermitian: Psi^dag = Psi
// For Pauli-Z: Psi is real diagonal -> trivially Hermitian.
// Exact values: Psi[0][0]=2, Psi[3][3]=-2.
// ---------------------------------------------------------------------------

TEST(HeomConstruction, Psi_Hermitian) {
    auto V   = make_pauli_z();
    auto Psi = make_Psi(V);

    EXPECT_NEAR(entry(Psi, 0, 0).real(), 2.0, EPS);
    EXPECT_NEAR(entry(Psi, 0, 0).imag(), 0.0, EPS);
    EXPECT_NEAR(entry(Psi, 3, 3).real(),-2.0, EPS);
    EXPECT_NEAR(entry(Psi, 3, 3).imag(), 0.0, EPS);

    // Mid-diagonal entries should be zero
    EXPECT_NEAR(std::abs(entry(Psi, 1, 1)), 0.0, EPS);
    EXPECT_NEAR(std::abs(entry(Psi, 2, 2)), 0.0, EPS);

    // Psi^dag = Psi
    lil_matrix<dynamic, c128, row_major, nil> Psi_adj;
    Psi_adj.set_adjoint(Psi);

    EXPECT_NEAR(entry(Psi_adj, 0, 0).real(), entry(Psi, 0, 0).real(), EPS);
    EXPECT_NEAR(entry(Psi_adj, 3, 3).real(), entry(Psi, 3, 3).real(), EPS);
}

// ---------------------------------------------------------------------------
// L = i*[H, .]  preserves trace: all column sums are zero.
// For Pauli-X: L has 8 non-zero entries, each column sums to 0.
// ---------------------------------------------------------------------------

TEST(HeomConstruction, L_TracePreserving) {
    auto H = make_pauli_x();
    auto L = make_L(H);

    // Column sums: for each column j, sum L[i][j] over all i
    std::map<int, c128> col_sum;
    for (auto& row_entry : L.data) {
        for (auto& col_entry : row_entry.second) {
            col_sum[col_entry.first] += col_entry.second;
        }
    }
    for (int j = 0; j < 4; ++j) {
        auto it = col_sum.find(j);
        c128 s = (it != col_sum.end()) ? it->second : c128{0,0};
        EXPECT_NEAR(s.real(), 0.0, EPS) << "column " << j;
        EXPECT_NEAR(s.imag(), 0.0, EPS) << "column " << j;
    }
}

// ---------------------------------------------------------------------------
// Xi = -s_delta * Phi^2 is Hermitian.
// For Pauli-Z: Phi^2 = diag(0,-4,-4,0), so Xi = s_delta*diag(0,4,4,0).
// This is real diagonal -> Hermitian.
// ---------------------------------------------------------------------------

TEST(HeomConstruction, Xi_Hermitian) {
    auto V           = make_pauli_z();
    auto Phi         = make_Phi(V);
    const double s_delta = 0.5;

    lil_matrix<dynamic, c128, row_major, nil> Xi;
    Xi.set_shape(4, 4);
    gemm<dynamic>(nilobj, c128{-s_delta, 0}, Phi, Phi, zero<c128>(), Xi, 4);
    Xi.optimize();

    // Expected: Xi[1][1] = 2, Xi[2][2] = 2
    EXPECT_NEAR(entry(Xi, 1, 1).real(), 2.0, EPS);
    EXPECT_NEAR(entry(Xi, 1, 1).imag(), 0.0, EPS);
    EXPECT_NEAR(entry(Xi, 2, 2).real(), 2.0, EPS);
    EXPECT_NEAR(entry(Xi, 2, 2).imag(), 0.0, EPS);
    EXPECT_NEAR(std::abs(entry(Xi, 0, 0)), 0.0, EPS);
    EXPECT_NEAR(std::abs(entry(Xi, 3, 3)), 0.0, EPS);

    // Xi^dag = Xi (diagonal real -> trivially Hermitian)
    lil_matrix<dynamic, c128, row_major, nil> Xi_adj;
    Xi_adj.set_adjoint(Xi);
    EXPECT_NEAR(entry(Xi_adj, 1, 1).real(), entry(Xi, 1, 1).real(), EPS);
    EXPECT_NEAR(entry(Xi_adj, 2, 2).real(), entry(Xi, 2, 2).real(), EPS);
}

// ---------------------------------------------------------------------------
// R_0 = L + Xi has the correct diagonal entries.
// For H=Pauli-X, V=Pauli-Z, s_delta=0.5:
//   L[1][1]=0, L[2][2]=0, Xi[1][1]=2, Xi[2][2]=2 -> R_0[1][1]=2, R_0[2][2]=2
//   L[0][0]=0, L[3][3]=0, Xi diag-0=0, Xi diag-3=0 -> R_0[0][0]=0, R_0[3][3]=0
// ---------------------------------------------------------------------------

TEST(HeomConstruction, R0_Diagonal) {
    auto H = make_pauli_x();
    auto V = make_pauli_z();
    const double s_delta = 0.5;

    auto L   = make_L(H);
    auto Phi = make_Phi(V);

    lil_matrix<dynamic, c128, row_major, nil> Xi;
    Xi.set_shape(4, 4);
    gemm<dynamic>(nilobj, c128{-s_delta, 0}, Phi, Phi, zero<c128>(), Xi, 4);

    lil_matrix<dynamic, c128, row_major, nil> R0;
    R0.set_shape(4, 4);
    axpy<dynamic>(nilobj, one<c128>(), L,  R0, 4);
    axpy<dynamic>(nilobj, one<c128>(), Xi, R0, 4);
    R0.optimize();

    // Off-Liouvillian-diagonal: from L (imaginary); diagonal: from Xi (real)
    EXPECT_NEAR(entry(R0, 1, 1).real(), 2.0, EPS);
    EXPECT_NEAR(entry(R0, 1, 1).imag(), 0.0, EPS);
    EXPECT_NEAR(entry(R0, 2, 2).real(), 2.0, EPS);
    EXPECT_NEAR(entry(R0, 2, 2).imag(), 0.0, EPS);

    // Off-diagonal contributions from L (Pauli-X structure)
    // L[0][2] = i, L[2][0] = i, L[1][3] = i, L[3][1] = i
    EXPECT_NEAR(entry(R0, 0, 2).real(), 0.0, EPS);
    EXPECT_NEAR(entry(R0, 0, 2).imag(), 1.0, EPS);
    EXPECT_NEAR(entry(R0, 2, 0).real(), 0.0, EPS);
    EXPECT_NEAR(entry(R0, 2, 0).imag(), 1.0, EPS);
}

} // namespace
