/*
 * LibHEOM: Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#include <iostream>
#include <fstream>
#include <iomanip>

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "const.h"
#include "dense_matrix.h"
#include "csr_matrix.h"
#include "heom.h"
#include "redfield.h"

#include "gpu_info.h"
#ifdef SUPPORT_GPU_PARALLELIZATION
#  include "heom_gpu.h"
#  include "redfield_gpu.h"
#endif

namespace py = pybind11;
using namespace libheom;

template<typename T>
class coo_matrix {
public:
  int rows;
  int cols;
  int nnz;
  py::array_t<int> row;
  py::array_t<int> col;
  py::array_t<T>   data;
  coo_matrix(int rows, int cols, int nnz,
            py::array_t<int> row, py::array_t<int> col, py::array_t<T> data) :
      rows(rows), cols(cols), nnz(nnz), row(row), col(col), data(data) {}
  void dump(lil_matrix<T>& mat) {
    mat.clear();
    const int* r = row.data();
    const int* c = col.data();
    const T* d   = data.data();
    mat.set_shape(rows, cols);
    for (int i = 0; i < nnz; ++i) {
      mat.data[r[i]][c[i]] = d[i];
    }
  }
  void dump(Eigen::SparseMatrix<T, Eigen::RowMajor>& mat) {
    std::vector<Eigen::Triplet<T>> list;
    const int* r = row.data();
    const int* c = col.data();
    const T* d   = data.data();
    mat.resize(rows, cols);
    for (int i = 0; i < nnz; ++i) {
      list.push_back(Eigen::Triplet<T>(r[i],c[i],d[i]));
    }
    mat.setFromTriplets(list.begin(), list.end());
    mat.makeCompressed();
  }
};


// #include "dok_matrix.h"
template<template <typename,
                   template <typename, int> class,
                   int> class qme_type,
         typename T,
         template<typename, int> class matrix_type,
         int num_state>
void set_hamiltonian(qme_type<T, matrix_type, num_state>& obj,
                     coo_matrix<T>& H) {
  if (H.rows != H.cols) {
    throw std::runtime_error("Hamiltonian must be a square matrix");
  }
  H.dump(obj.H);
  obj.n_state = H.rows;
}


template<template <typename,
                   template <typename, int> class,
                   int> class qme_type,
         typename T,
         template<typename, int> class matrix_type, 
         int num_state>
void allocate_noises(qme_type<T, matrix_type, num_state>& obj,
                     int n_noise) {
  obj.allocate_noise(n_noise);
}


template<template <typename,
                   template <typename, int> class,
                   int> class qme_type,
         typename T,
         template<typename, int> class matrix_type, 
         int num_state>
void set_noise(qme_type<T, matrix_type, num_state>& obj,
               int u,
               coo_matrix<T>& V,
               coo_matrix<T>& gamma,
               py::array_t<T> phi_0,
               py::array_t<T> sigma,
               coo_matrix<T>& s,
               T S_delta,
               coo_matrix<T>& a) {
  if (V.rows != V.cols) {
    throw std::runtime_error("[Error] Noise operator must be a square matrix");
  }
  if (V.rows != obj.n_state) {
    throw std::runtime_error("[Error] Hamiltonian and noise operators must have the same dimension");
  }
  
  V.dump(obj.V[u]);
  
  obj.len_gamma[u] = static_cast<int>(gamma.rows);
  gamma.dump(obj.gamma[u]);
  obj.phi_0[u].resize(phi_0.shape(0));
  std::copy_n(phi_0.data(), phi_0.shape(0), obj.phi_0[u].data());
  obj.sigma[u].resize(obj.len_gamma[u]);
  std::copy_n(sigma.data(), sigma.shape(0), obj.sigma[u].data());
  s.dump(obj.s[u]);
  obj.S_delta[u] = S_delta;
  a.dump(obj.a[u]);
}


template<template <typename,
                   template <typename, int> class,
                   int> class qme_type,
         typename T,
         template<typename, int> class matrix_type, 
         int num_state>
void set_noise_func(qme_type<T, matrix_type, num_state>& obj,
                    int u,
                    coo_matrix<T>& V,
                    std::function<T(REAL_TYPE(T))> corr) {
  if (V.rows != V.cols) {
    throw std::runtime_error("[Error] Noise operator must be a square matrix");
  }
  if (V.rows != obj.n_state) {
    throw std::runtime_error("[Error] Hamiltonian and noise operators must have the same dimension");
  }
  
  V.dump(obj.V[u]);
  
  if (!obj.use_corr_func) {
    obj.use_corr_func.reset(new bool [obj.n_noise]);
    std::fill_n(&obj.use_corr_func[0], obj.n_noise, false);
    obj.corr_func.reset(new std::function<T(REAL_TYPE(T))> [obj.n_noise]);
  }
  obj.use_corr_func[u] = true;
  obj.corr_func[u] = corr;
  
}


template<template <typename,
                   template <typename, int> class,
                   int> class heom_type,
         typename T,
         template<typename, int> class matrix_type, 
         int num_state>
void flatten_hierarchy_dimension(heom_type<T, matrix_type, num_state>& obj) {
  obj.linearize_dim();
}



// void set_dt
// /**/(float64 dt__unit, float64 dt)
// {
//   params::h.dt__unit = dt__unit;
//   params::h.dt       = dt;
// }


template<template <typename,
                   template <typename, int> class,
                   int> class heom_type,
         typename T,
         template<typename, int> class matrix_type, 
         int num_state>
int allocate_hierarchy_space(heom_type<T, matrix_type, num_state>& obj,
                             int max_depth,
                             py::function& callback,
                             int interval_callback) {
  obj.n_hierarchy
      = allocate_hierarchy_space(obj.hs,
                                 max_depth,
                                 [&](double progress) {
                                   // std::cout << ":progress " << progress << std::endl;
                                   // callback(progress);
                                 },
                                 interval_callback);
  return obj.n_hierarchy + 1;
}


template<template <typename,
                   template <typename, int> class,
                   int> class qme_type,
         typename T,
         template<typename, int> class matrix_type, 
         int num_state>
void init_aux_vars(qme_type<T, matrix_type, num_state>& obj,
                   py::function& callback,
                   int interval_callback) {
  obj.initialize();
  obj.init_aux_vars([&](int lidx) { callback(lidx); });
}


// template<template <typename, template <typename> class> class qme_type, typename T>
// void construct_commutator
// /**/(qme_type<T>& obj,
//      int shape_0,
//      int shape_1,
//      int nnz,
//      py::array_t<int> row,
//      py::array_t<int> col,
//      py::array_t<T> data,
//      T coef_l,
//      T coef_r,
//      py::function& callback,
//      int interval_callback)
// {
//   lil_matrix<T> x;
//   const int* r = row.data();
//   const int* c = col.data();
//   const T* d = data.data();
//   std::get<0>(x.shape) = shape_0;
//   std::get<1>(x.shape) = shape_1;
//   for (int i = 0; i < nnz; ++i) {
//     x.data[r[i]][c[i]] = d[i];
//   }
//   obj.construct_commutator(x,
//                            coef_l,
//                            coef_r,
//                            [&](int lidx) { callback(lidx); },
//                            interval_callback);
// }


// template<template <typename, template <typename> class> class qme_type, typename T>
// void apply_commutator
// /**/(qme_type<T>& obj,
//      py::array_t<T> rho)
// {
//   obj.apply_commutator(rho.mutable_data());
// }

template<template <typename,
                   template <typename, int> class,
                   int> class qme_type,
         typename T,
         template<typename, int> class matrix_type, 
         int num_state>
void calc_diff(qme_type<T, matrix_type, num_state>& obj,
               py::array_t<T> drho_dt,
               const py::array_t<T> rho,
               REAL_TYPE(T) alpha,
               REAL_TYPE(T) beta) {
  obj.calc_diff(Eigen::Map<dense_vector<T,Eigen::Dynamic>>(drho_dt.mutable_data(), obj.size_rho),
                Eigen::Map<const dense_vector<T,Eigen::Dynamic>>(rho.data(), obj.size_rho),
                alpha,
                beta);
}


template<template <typename,
                   template <typename, int> class,
                   int> class qme_type,
         typename T,
         template<typename, int> class matrix_type, 
         int num_state>
void time_evolution(qme_type<T, matrix_type, num_state>& obj,
                    py::array_t<T> rho,
                    typename T::value_type dt__unit,
                    typename T::value_type dt,
                    int interval,
                    int count,
                    py::function& callback) {
  obj.time_evolution(Eigen::Map<dense_vector<T,Eigen::Dynamic>>(rho.mutable_data(), obj.size_rho),
                     dt__unit, dt,
                     interval,
                     count,
                     [&](typename T::value_type t) { callback(t); });
}


template<template <typename,
                   template <typename, int> class,
                   int> class qme_type,
         typename T,
         template<typename, int> class matrix_type, 
         int num_state>
void set_device_number(qme_type<T, matrix_type, num_state>& obj,
                       int device_number) {
  obj.device_number = device_number;
}

// std::vector<std::vector<int>>& get_lk
// /**/(hierarchy& h)
// {
//   return h.lk; 
// }

// std::vector<std::vector<int>>& get_j
// /**/(heom_liou_hrchy_liou<complex128>& h)
// {
//   return h.hs.j; 
// }


template<template <typename,
                   template <typename, int> class,
                   int> class qme_type,
         typename T,
         template<typename, int> class matrix_type, 
         int num_state>
py::class_<qme_type<T, matrix_type, num_state>> declare_qme_binding(
    py::module m,
    const char* class_name) {
  return py::class_<qme_type<T, matrix_type, num_state>>(m, class_name)
      .def(py::init<>())
      .def("set_hamiltonian",      &set_hamiltonian<qme_type, T, matrix_type, num_state>)
      .def("allocate_noises",      &allocate_noises<qme_type, T, matrix_type, num_state>)
      .def("set_noise",            &set_noise      <qme_type, T, matrix_type, num_state>)
      .def("init_aux_vars",        &init_aux_vars  <qme_type, T, matrix_type, num_state>)
      .def("calc_diff",            &calc_diff      <qme_type, T, matrix_type, num_state>)
      .def("time_evolution",       &time_evolution <qme_type, T, matrix_type, num_state>);
      // .def("construct_commutator", &construct_commutator <qme_type, T, matrix_type>)
      // .def("apply_commutator",     &apply_commutator     <qme_type, T, matrix_type>)
}


template<template <typename,
                   template <typename, int> class,
                   int> class redfield_type,
         typename T,
         template<typename, int> class matrix_type, 
         int num_state>
py::class_<redfield_type<T, matrix_type, num_state>> declare_redfield_binding(
    py::module m,
    const char* class_name)
{
  return declare_qme_binding<redfield_type, T, matrix_type, num_state>(m, class_name)
      .def("set_noise_func", &set_noise_func <redfield_type, T, matrix_type, num_state>);
}


template<template <typename,
                   template <typename, int> class,
                   int> class heom_type,
         typename T,
         template<typename, int> class matrix_type, 
         int num_state>
py::class_<heom_type<T, matrix_type, num_state>> declare_heom_binding(
    py::module m,
    const char* class_name)
{
  return declare_qme_binding<heom_type, T, matrix_type, num_state>(m, class_name)
      .def("flatten_hierarchy_dimension", &flatten_hierarchy_dimension<heom_type, T, matrix_type, num_state>)
      .def("allocate_hierarchy_space",    &allocate_hierarchy_space   <heom_type, T, matrix_type, num_state>);
}


template<template <typename,
                   template <typename, int> class,
                   int> class qme_type,
         typename T,
         template<typename, int> class matrix_type, 
         int num_state>
py::class_<qme_type<T, matrix_type, num_state>> declare_qme_gpu_binding(
    py::module m,
    const char* class_name) {
  return py::class_<qme_type<T, matrix_type, num_state>>(m, class_name)
      .def(py::init<>())
      .def("set_hamiltonian",         &set_hamiltonian        <qme_type, T, matrix_type, num_state>)
      .def("allocate_noises",             &allocate_noises            <qme_type, T, matrix_type, num_state>)
      .def("set_noise",                   &set_noise              <qme_type, T, matrix_type, num_state>)
      .def("init_aux_vars",               &init_aux_vars              <qme_type, T, matrix_type, num_state>)
      // .def("construct_commutator",        &construct_commutator       <qme_type, T, matrix_type>)
      // .def("apply_commutator",            &apply_commutator           <qme_type, T, matrix_type>)
      .def("time_evolution",              &time_evolution             <qme_type, T, matrix_type, num_state>)
      .def("set_device_number",           &set_device_number          <qme_type, T, matrix_type, num_state>)
      .def("set_noise_func",    &set_noise_func    <qme_type, T, matrix_type, num_state>);
}


template<template <typename,
                   template <typename, int> class,
                   int> class heom_type,
         typename T,
         template<typename, int> class matrix_type, 
         int num_state>
py::class_<heom_type<T, matrix_type, num_state>> declare_heom_gpu_binding(
    py::module m,
    const char* class_name) {
  return py::class_<heom_type<T, matrix_type, num_state>>(m, class_name)
      .def(py::init<>())
      .def("set_hamiltonian",         &set_hamiltonian        <heom_type, T, matrix_type, num_state>)
      .def("allocate_noises",             &allocate_noises            <heom_type, T, matrix_type, num_state>)
      .def("set_noise",               &set_noise              <heom_type, T, matrix_type, num_state>)
      .def("init_aux_vars",               &init_aux_vars              <heom_type, T, matrix_type, num_state>)
      // .def("construct_commutator",        &construct_commutator       <heom_type, T, matrix_type>)
      // .def("apply_commutator",            &apply_commutator           <heom_type, T, matrix_type>)
      .def("time_evolution",              &time_evolution             <heom_type, T, matrix_type, num_state>)
      .def("flatten_hierarchy_dimension", &flatten_hierarchy_dimension<heom_type, T, matrix_type, num_state>)
      .def("allocate_hierarchy_space",    &allocate_hierarchy_space   <heom_type, T, matrix_type, num_state>)
      .def("set_device_number",           &set_device_number          <heom_type, T, matrix_type, num_state>);
}


PYBIND11_MODULE(pylibheom, m) {
  m.doc() = "Low-level Python Binding of LibHEOM";

  m.attr("support_gpu_parallelization")
      = py::cast(support_gpu_parallelization);
  m.def("gpu_device_count", &get_gpu_device_count);
  m.def("gpu_device_name",  &get_gpu_device_name);

  py::class_<coo_matrix<complex64 >>(m, "coo_matrix_c")
      .def(py::init<int, int, int,
           py::array_t<int>, py::array_t<int>, py::array_t<complex64>>());
  py::class_<coo_matrix<complex128>>(m, "coo_matrix_z")
      .def(py::init<int, int, int,
           py::array_t<int>, py::array_t<int>, py::array_t<complex128>>());
    
  declare_heom_binding<heom_ll, complex128, dense_matrix, Eigen::Dynamic>(m, "heom_zdll");
  declare_heom_binding<heom_ll, complex128, csr_matrix,   Eigen::Dynamic>(m, "heom_zsll");
  declare_heom_binding<heom_lh, complex128, dense_matrix, Eigen::Dynamic>(m, "heom_zdlh");
  declare_heom_binding<heom_lh, complex128, csr_matrix,   Eigen::Dynamic>(m, "heom_zslh");
  declare_redfield_binding<redfield_h, complex128, dense_matrix, Eigen::Dynamic>(m, "redfield_zdh");
  declare_redfield_binding<redfield_h, complex128, csr_matrix,   Eigen::Dynamic>(m, "redfield_zsh");
  declare_redfield_binding<redfield_l, complex128, dense_matrix, Eigen::Dynamic>(m, "redfield_zdl");
  declare_redfield_binding<redfield_l, complex128, csr_matrix,   Eigen::Dynamic>(m, "redfield_zsl");
  // declare_heom_binding<heom_ll,    complex128, dense_matrix, 2>(m, "heom_zdll");
  // declare_heom_binding<heom_ll,    complex128, csr_matrix,   2>(m, "heom_zsll");
  // declare_heom_binding<heom_lh,    complex128, dense_matrix, 2>(m, "heom_zdlh");
  // declare_heom_binding<heom_lh,    complex128, csr_matrix,   2>(m, "heom_zslh");
  // declare_qme_binding <redfield_h, complex128, dense_matrix, 2>(m, "redfield_zdh");
  // declare_qme_binding <redfield_h, complex128, csr_matrix,   2>(m, "redfield_zsh");
  // declare_qme_binding <redfield_l, complex128, dense_matrix, 2>(m, "redfield_zdl");
  // declare_qme_binding <redfield_l, complex128, csr_matrix,   2>(m, "redfield_zsl");
#ifdef SUPPORT_GPU_PARALLELIZATION
  declare_heom_gpu_binding<heom_lh_gpu,   complex128, dense_matrix, Eigen::Dynamic>(m, "heom_zdlh_gpu");
  declare_heom_gpu_binding<heom_lh_gpu,   complex128, csr_matrix,   Eigen::Dynamic>(m, "heom_zslh_gpu");
  declare_redfield_gpu_binding <redfield_h_gpu,complex128, dense_matrix, Eigen::Dynamic>(m, "redfield_zdh_gpu");
  declare_redfield_gpu_binding <redfield_h_gpu,complex128, csr_matrix,   Eigen::Dynamic>(m, "redfield_zsh_gpu");
  // declare_qme_gpu_binding <redfield_l_gpu,complex128, dense_matrix, Eigen::Dynamic>(m, "redfield_zdl_gpu");
  // declare_qme_gpu_binding <redfield_l_gpu,complex128, csr_matrix,   Eigen::Dynamic>(m, "redfield_zsl_gpu");
#endif
}
