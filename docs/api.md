# API Reference

libheom is a C++17 template library. All classes live in the `libheom` namespace.
Include the single top-level header:

```cpp
#include "libheom.h"
```

## Types

```cpp
using float32   = float;
using float64   = double;
using complex64  = std::complex<float>;
using complex128 = std::complex<double>;
```

`real_t<dtype>` gives the real scalar type for a complex dtype
(`float` for `complex64`, `double` for `complex128`).

## Template parameters

Most classes share the same set of template parameters:

- `n_level_c` - compile-time system size; use `-1` for dynamic (runtime) sizing
- `dtype` - scalar type: `complex64` or `complex128`
- `matrix_base` - matrix storage: `dense_matrix` or `sparse_matrix`
- `order` - memory layout: `row_major` or `col_major`
- `linalg_engine` - backend: `linalg_engine_eigen`, `linalg_engine_mkl`, or `linalg_engine_cuda`

## Linear algebra engines

```cpp
linalg_engine_eigen  engine_eigen;
linalg_engine_mkl    engine_mkl;
linalg_engine_cuda   engine_cuda;
```

Each engine manages memory and BLAS/LAPACK calls for its backend.
Construct one engine per solver instance and pass a pointer to the `qme_solver`.

## Equation-of-motion classes

### HEOM

Three space representations are available:

```cpp
// Hilbert space (rho is n_level x n_level)
heom_hilb<n_level_c, dtype, matrix_base, order, linalg_engine>

// Liouville space (rho is n_level^2 vector)
heom_liou<n_level_c, dtype, matrix_base, order, order_liou, linalg_engine>

// ADO space (operates directly on the full hierarchy vector)
heom_ado<n_level_c, dtype, matrix_base, order, linalg_engine>
```

All three inherit from `heom<dtype, order, linalg_engine>` which inherits from
`qme_base<dtype, order, linalg_engine>`.

Constructor (all representations):

```cpp
heom_hilb<...> qme(int truncation_depth, int n_inner_threads, int n_outer_threads);
```

Key fields set by the caller before `set_param` (all inherited from `qme_base`
or `heom`):

```cpp
int    n_level;         // system dimension (set before alloc_noises)
lil_matrix<...> H;     // Hamiltonian

// Call alloc_noises(n_noise) first, then for each noise source u:
lil_matrix<...> V[u];        // system-bath coupling operator
lil_matrix<...> gamma[u];    // exponential decay matrix (K x K)
vector<dtype>   phi_0[u];    // initial vector of phi(t)=exp(-gamma t)*phi_0, on the right of C(t) (length K)
vector<dtype>   sigma[u];    // constant coefficient vector, applied as sigma^T on the left of C(t) (length K)
lil_matrix<...> s_mat[u];    // real part of C(t): S in C(t)=sigma^T (s_mat+i*a_mat) exp(-gamma t) phi_0
lil_matrix<...> a_mat[u];    // imaginary part of C(t)
dtype           s_delta[u];  // delta-function weight
```

Key methods:

```cpp
void alloc_noises(int n_noise);          // allocate noise arrays
void set_param(linalg_engine* engine);   // finalize setup, upload to engine memory
int  get_n_hierarchy();                  // number of hierarchy nodes (call after set_param)
```

`qme_solver` also exposes a time derivative call:

```cpp
void calc_time_derivative(dtype* drho_dt, dtype* rho);
```

### Redfield

```cpp
redfield_hilb<n_level_c, dtype, matrix_base, order, linalg_engine>
redfield_liou<n_level_c, dtype, matrix_base, order, order_liou, linalg_engine>
```

Same interface as the HEOM classes; no hierarchy dimension (`n_hierarchy == 1`).

## Solver classes

Solvers are passed to `qme_solver` and perform time integration.

| Class | Type | Description |
|---|---|---|
| `rk4<dtype, order, linalg_engine>` | fixed step | 4th-order Runge-Kutta |
| `lsrk4<dtype, order, linalg_engine>` | fixed step | low-storage RK4 (less memory) |
| `rkdp<dtype, order, linalg_engine>` | adaptive step | Dormand-Prince RK45 |

Adaptive solvers accept `atol` and `rtol` in the `kwargs` map passed to `solve()`.

## `qme_solver`

The `qme_solver` class ties together an engine, an equation of motion, and a solver:

```cpp
template<typename dtype, order_t order, typename linalg_engine>
class qme_solver {
public:
  qme_solver(linalg_engine* engine,
             qme_base<dtype,order,linalg_engine>* qme,
             solver_base<dtype,order,linalg_engine>* solver);

  void solve(dtype* rho,
             const real_t<dtype>* t_list,
             int n_t,
             std::function<void(real_t<dtype>)> callback,
             const kwargs_t& kwargs);
};
```

`rho` is the flattened density matrix / hierarchy array (size `main_size()`).
`t_list` is an array of `n_t` output times.
`callback` is called with the current time at each output step.
`kwargs` is a `std::map<std::string, std::any>` for solver-specific options
(e.g. `{"dt", 0.0025}` for fixed-step solvers, `{"atol", 1e-8}` for rkdp).

## Usage example

```cpp
#include "libheom.h"
using namespace libheom;

// Engine (Eigen, double precision, row-major)
using Engine = linalg_engine_eigen;
using dtype  = complex128;
constexpr order_t Ord = row_major;

// Build the HEOM object
heom_hilb<-1, dtype, dense_matrix, Ord, Engine> qme(
    /*truncation_depth=*/5, /*n_inner=*/1, /*n_outer=*/1);

qme.n_level = 2;
qme.H.set_shape(2, 2);
// ... fill qme.H ...

qme.alloc_noises(1);
qme.gamma[0].set_shape(1, 1);
qme.gamma[0].push(0, 0, {1.0, 0.0});   // single Drude pole
qme.phi_0[0] = {1.0};
qme.sigma[0] = {1.0};
qme.s_mat[0].set_shape(1, 1);
qme.s_mat[0].push(0, 0, {0.1, 0.0});   // real coupling coefficient
qme.a_mat[0].set_shape(1, 1);
qme.a_mat[0].push(0, 0, {0.0, 0.0});   // imaginary coupling coefficient
qme.s_delta[0] = 0.0;
qme.V[0].set_shape(2, 2);
// ... fill qme.V[0] ...

Engine engine;
qme.set_param(&engine);

// Solver
lsrk4<dtype, Ord, Engine> solver;
qme_solver<dtype, Ord, Engine> runner(&engine, &qme, &solver);

// Time evolution
std::vector<double> t_list = {0.0, 1.0, 2.0, 5.0};
kwargs_t kwargs = {{"dt", 0.0025}};
runner.solve(rho_data, t_list.data(), t_list.size(), callback, kwargs);
```

In practice, pyheom's `pylibheom` extension pre-instantiates all commonly used
template combinations and exposes them through a Python API; direct C++ use is
only needed for embedding libheom in other C++ projects.
