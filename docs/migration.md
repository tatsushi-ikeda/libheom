# Migration Guide

## v0.5 (SI) -> v1.0

This guide covers breaking changes for users who wrote C++ code directly against
libheom v0.5 (the supplementary information release accompanying the JCP 2020 paper).

### Naming convention

All names changed from PascalCase/CamelCase to snake_case.

```cpp
// v0.5
HierarchySpace   AllocateHierarchySpace   CalcDiff   InitAuxVars

// v1.0
hierarchy_space  alloc_hierarchy_space    calc_diff  set_param
```

### Entry-point header

```cpp
// v0.5
#include "heom.h"
#include "redfield.h"

// v1.0
#include "libheom.h"   // single top-level header; include nothing else
```

### Class renames

The two-letter suffix encodes the EOM space (H = Hilbert, L = Liouville).
ADO space is new in v1.0 and has no v0.5 equivalent.

```cpp
// v0.5                                     v1.0
HierarchySpace                           -> hierarchy_space
Qme<T>                                   -> qme_base<dtype, order, linalg_engine>
Heom<T>                                  -> heom<dtype, order, linalg_engine>
HeomLH<T, MatrixType, NumState>          -> heom_hilb<n_level_c, dtype, matrix_base,
                                                       order, linalg_engine>
HeomLL<T, MatrixType, NumState>          -> heom_liou<n_level_c, dtype, matrix_base,
                                                       order, order_liou, linalg_engine>
// (new in v1.0)                          -> heom_ado<n_level_c, dtype, matrix_base,
                                                       order, linalg_engine>
Redfield<T>                              -> redfield<dtype, order, linalg_engine>
RedfieldH<T, MatrixType, NumState>       -> redfield_hilb<n_level_c, dtype, matrix_base,
                                                           order, linalg_engine>
RedfieldL<T, MatrixType, NumState>       -> redfield_liou<n_level_c, dtype, matrix_base,
                                                           order, order_liou, linalg_engine>

// GPU (v0.5 separate classes)            // v1.0: use linalg_engine_cuda as template arg
HeomLHGpu<T, MatrixType, NumState>       -> heom_hilb<..., linalg_engine_cuda>
RedfieldHGpu<T, MatrixType, NumState>    -> redfield_hilb<..., linalg_engine_cuda>
```

### Template parameters

```cpp
// v0.5
template<typename T,
         template<typename, int> class MatrixType,
         int NumState>
class HeomLH : public HeomL<T>

// v1.0
template<int n_level_c,            // compile-time system size; use -1 for dynamic
         typename dtype,           // was T
         template<int, typename,
                  order_t,
                  typename> class matrix_base,   // was MatrixType
         order_t order,            // new: row_major or col_major
         typename linalg_engine>   // new: selects backend (Eigen / MKL / CUDA)
class heom_hilb : public heom<dtype, order, linalg_engine>
```

Concrete template argument examples:

```cpp
// v0.5
HeomLH<complex<double>, DenseMatrix, Eigen::Dynamic> qme;
HeomLH<complex<double>, CsrMatrix,   Eigen::Dynamic> qme;
HeomLHGpu<complex<double>, DenseMatrix, Eigen::Dynamic> qme;

// v1.0
heom_hilb<-1, complex128, dense_matrix,  row_major, linalg_engine_eigen> qme(...);
heom_hilb<-1, complex128, sparse_matrix, row_major, linalg_engine_eigen> qme(...);
heom_hilb<-1, complex128, dense_matrix,  row_major, linalg_engine_cuda>  qme(...);
```

### Matrix type renames

```cpp
// v0.5                                   // v1.0 (template alias)
DenseMatrix<T,N>  (Eigen column-major)  -> dense_matrix<n_level_c, dtype, order, engine>
CsrMatrix<T>      (Eigen CSR)           -> sparse_matrix<n_level_c, dtype, order, engine>
LilMatrix<T>      (list-of-lists)       -> lil_matrix<dynamic, dtype, order, nil>
```

### Member renames

In `Qme` / `qme_base`:

```cpp
// v0.5     v1.0
n_state   -> n_level
s         -> S          // S_mat: system-bath coupling real part (lil_matrix)
a         -> A          // A_mat: system-bath coupling imaginary part (lil_matrix)
S_delta   -> s_delta    // delta-function weight
```

In `Heom` / `heom`:

```cpp
// v0.5           v1.0
jgamma_diag    -> n_gamma_diag   // sum_k n_k * gamma_kk per hierarchy node
n_dim          -> (hierarchy_space::n_modes)  // moved into hierarchy_space
S   (vectors)  -> s              // per-mode amplitude vectors
A   (vectors)  -> a
```

In `HierarchySpace` / `hierarchy_space`:

```cpp
// v0.5         v1.0
n_dim         -> n_modes         // total number of exponential modes
j             -> n               // multi-index vectors
index_book    -> multi_index_map // map from multi-index to linear index
```

In Redfield implementation:

```cpp
// v0.5                   v1.0 (redfield_hilb::impl)
Lambda_dagger_impl      -> Lambda_dag
```

### Function renames

```cpp
// v0.5
AllocateHierarchySpace(hs, max_depth, callback, interval, filter_predicator);

// v1.0
alloc_hierarchy_space(hs, truncation_depth,       // max_depth -> truncation_depth
                      callback,                    // signature: void(int,int) not void(double)
                      interval,
                      hierarchy_filter,            // filter_predicator -> hierarchy_filter
                      filter_flag);                // new bool parameter
```

```cpp
// v0.5
obj.AllocateNoise(n_noise);

// v1.0
qme.alloc_noises(n_noise);    // note plural "noises"
```

### Setup and time-evolution API

```cpp
// v0.5 setup
HeomLH<complex<double>, DenseMatrix, Eigen::Dynamic> qme;
qme.n_state = n;
qme.H = ...;
qme.AllocateNoise(n_noise);
// ... fill qme.gamma[u], qme.phi_0[u], qme.sigma[u], qme.s[u], qme.S_delta[u], qme.a[u] ...
qme.V[u] = ...;
AllocateHierarchySpace(qme.hs, max_depth);
qme.InitAuxVars([](int lidx) {});    // main setup call

// v1.0 setup
linalg_engine_eigen engine;
heom_hilb<-1, complex128, dense_matrix, row_major, linalg_engine_eigen>
    qme(truncation_depth, n_inner_threads, n_outer_threads);
qme.n_level = n;
qme.H = ...;
qme.alloc_noises(n_noise);
// ... fill qme.gamma[u], qme.phi_0[u], qme.sigma[u], qme.S[u], qme.s_delta[u], qme.A[u] ...
qme.V[u] = ...;
qme.set_param(&engine);    // allocates hierarchy + uploads to engine
```

```cpp
// v0.5 time evolution
qme.TimeEvolution(rho, dt_unit, dt, interval, count, callback);

// v1.0 time evolution
lsrk4<complex128, row_major, linalg_engine_eigen> solver;
qme_solver<complex128, row_major, linalg_engine_eigen> runner(&engine, &qme, &solver);

std::vector<double> t_list = {0.0, 1.0, 2.0, 5.0};
kwargs_t kwargs = {{"dt", 0.0025}};
runner.solve(rho_data, t_list.data(), t_list.size(), callback, kwargs);
```

The flat `rho_data` array has `main_size()` elements:
- Hilbert space: `n_level * n_level` (single density matrix)
- Liouville / Hilbert HEOM: `n_level * n_level * n_hierarchy`
- ADO space: `n_level * n_hierarchy` (not squared)

### Build system

v0.5 required a separate build directory for each backend:

```
libheom_0.5/build_gcc_eigen/        (CPU/Eigen)
libheom_0.5/build_intel_eigen_mkl/  (CPU/MKL)
libheom_0.5/build_gcc_cuda/         (GPU/CUDA)
```

v1.0 builds all enabled backends into a single library:

```bash
cmake -DLIBHEOM_USE_MKL=ON -DLIBHEOM_ENABLE_CUDA=ON -DCUDA_ARCH_LIST=70 ..
make
```

The backend is selected at compile time via the `linalg_engine` template parameter,
not at link time.

### New in v1.0

- `heom_ado`: ADO-space representation.  Operates on the full hierarchy vector
  directly without constructing individual node-to-node coupling matrices.

- `qme_solver`: ties a `qme_base`, a `linalg_engine`, and a time integrator
  together.  Replaces the `TimeEvolution` member function.

- Adaptive integrator `rkdp` (Dormand-Prince RK45): pass `{"atol", 1e-8}` and
  `{"rtol", 1e-6}` in the `kwargs_t` map.

- `linalg_engine/` directory: a common interface for Eigen, MKL, and CUDA
  backends.  GPU classes (`HeomLHGpu`, `RedfieldHGpu`) are replaced by the
  same HEOM/Redfield templates instantiated with `linalg_engine_cuda`.
