/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
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
class CooMatrix {
public:
  int rows;
  int cols;
  int nnz;
  py::array_t<int> row;
  py::array_t<int> col;
  py::array_t<T>   data;
  CooMatrix(int rows, int cols, int nnz,
            py::array_t<int> row, py::array_t<int> col, py::array_t<T> data) :
      rows(rows), cols(cols), nnz(nnz), row(row), col(col), data(data) {}
  void Dump(LilMatrix<T>& mat) {
    mat.Clear();
    const int* r = row.data();
    const int* c = col.data();
    const T* d   = data.data();
    mat.SetShape(rows, cols);
    for (int i = 0; i < nnz; ++i) {
      mat.data[r[i]][c[i]] = d[i];
    }
  }
  void Dump(Eigen::SparseMatrix<T, Eigen::RowMajor>& mat) {
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
                   int> class QmeType,
         typename T,
         template<typename, int> class MatrixType,
         int NumState>
void SetHamiltonian(QmeType<T, MatrixType, NumState>& obj,
                    CooMatrix<T>& H) {
  if (H.rows != H.cols) {
    throw std::runtime_error("Hamiltonian must be a square matrix");
  }
  H.Dump(obj.H);
  obj.n_state = H.rows;
}


template<template <typename,
                   template <typename, int> class,
                   int> class QmeType,
         typename T,
         template<typename, int> class MatrixType, 
         int NumState>
void AllocateNoises(QmeType<T, MatrixType, NumState>& obj,
                     int n_noise) {
  obj.AllocateNoise(n_noise);
}


template<template <typename,
                   template <typename, int> class,
                   int> class QmeType,
         typename T,
         template<typename, int> class MatrixType, 
         int NumState>
void SetNoise(QmeType<T, MatrixType, NumState>& obj,
              int u,
              CooMatrix<T>& V,
              CooMatrix<T>& gamma,
              py::array_t<T> phi_0,
              py::array_t<T> sigma,
              CooMatrix<T>& s,
              T S_delta,
              CooMatrix<T>& a) {
  if (V.rows != V.cols) {
    throw std::runtime_error("[Error] Noise operator must be a square matrix");
  }
  if (V.rows != obj.n_state) {
    throw std::runtime_error("[Error] Hamiltonian and noise operators must have the same dimension");
  }
  
  V.Dump(obj.V[u]);
  
  obj.len_gamma[u] = static_cast<int>(gamma.rows);
  gamma.Dump(obj.gamma[u]);
  obj.phi_0[u].resize(phi_0.shape(0));
  std::copy_n(phi_0.data(), phi_0.shape(0), obj.phi_0[u].data());
  obj.sigma[u].resize(obj.len_gamma[u]);
  std::copy_n(sigma.data(), sigma.shape(0), obj.sigma[u].data());
  s.Dump(obj.s[u]);
  obj.S_delta[u] = S_delta;
  a.Dump(obj.a[u]);
}


template<template <typename,
                   template <typename, int> class,
                   int> class QmeType,
         typename T,
         template<typename, int> class MatrixType, 
         int NumState>
void SetNoiseFunc(QmeType<T, MatrixType, NumState>& obj,
                  int u,
                  CooMatrix<T>& V,
                  std::function<T(REAL_TYPE(T))> corr) {
  if (V.rows != V.cols) {
    throw std::runtime_error("[Error] Noise operator must be a square matrix");
  }
  if (V.rows != obj.n_state) {
    throw std::runtime_error("[Error] Hamiltonian and noise operators must have the same dimension");
  }
  
  V.Dump(obj.V[u]);
  
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
                   int> class HeomType,
         typename T,
         template<typename, int> class MatrixType, 
         int NumState>
void FlattenHierarchyDimension(HeomType<T, MatrixType, NumState>& obj) {
  obj.LinearizeDim();
}



// void set_dt
// /**/(float64 dt__unit, float64 dt)
// {
//   params::h.dt__unit = dt__unit;
//   params::h.dt       = dt;
// }


template<template <typename,
                   template <typename, int> class,
                   int> class HeomType,
         typename T,
         template<typename, int> class MatrixType, 
         int NumState>
int AllocateHierarchySpace(HeomType<T, MatrixType, NumState>& obj,
                           int max_depth,
                           py::function& callback,
                           int interval_callback) {
  obj.n_hierarchy
      = AllocateHierarchySpace(obj.hs,
                               max_depth,
                               [&](double progress) {
                                 std::cout << ":progress " << progress << std::endl;
                                 // callback(progress);
                               },
                               interval_callback);
  return obj.n_hierarchy + 1;
}


template<template <typename,
                   template <typename, int> class,
                   int> class QmeType,
         typename T,
         template<typename, int> class MatrixType, 
         int NumState>
void InitAuxVars(QmeType<T, MatrixType, NumState>& obj,
                   py::function& callback,
                   int interval_callback) {
  obj.Initialize();
  obj.InitAuxVars([&](int lidx) { callback(lidx); });
}


// template<template <typename, template <typename> class> class QmeType, typename T>
// void construct_commutator
// /**/(QmeType<T>& obj,
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
//   LilMatrix<T> x;
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


// template<template <typename, template <typename> class> class QmeType, typename T>
// void apply_commutator
// /**/(QmeType<T>& obj,
//      py::array_t<T> rho)
// {
//   obj.apply_commutator(rho.mutable_data());
// }


template<template <typename,
                   template <typename, int> class,
                   int> class QmeType,
         typename T,
         template<typename, int> class MatrixType, 
         int NumState>
void TimeEvolution(QmeType<T, MatrixType, NumState>& obj,
                    py::array_t<T> rho,
                    typename T::value_type dt__unit,
                    typename T::value_type dt,
                    int interval,
                    int count,
                    py::function& callback) {
  obj.TimeEvolution(Eigen::Map<DenseVector<T,Eigen::Dynamic>>(rho.mutable_data(), obj.size_rho),
                    dt__unit, dt,
                    interval,
                    count,
                    [&](typename T::value_type t) { callback(t); });
}


template<template <typename,
                   template <typename, int> class,
                   int> class QmeType,
         typename T,
         template<typename, int> class MatrixType, 
         int NumState>
void SetDeviceNumber(QmeType<T, MatrixType, NumState>& obj,
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
                   int> class QmeType,
         typename T,
         template<typename, int> class MatrixType, 
         int NumState>
py::class_<QmeType<T, MatrixType, NumState>> declare_qme_binding(py::module m, const char* class_name) {
  return py::class_<QmeType<T, MatrixType, NumState>>(m, class_name)
      .def(py::init<>())
      .def("set_Hamiltonian",         &SetHamiltonian        <QmeType, T, MatrixType, NumState>)
      .def("allocate_noises",         &AllocateNoises        <QmeType, T, MatrixType, NumState>)
      .def("set_noise",               &SetNoise              <QmeType, T, MatrixType, NumState>)
      .def("set_noise_func",          &SetNoiseFunc    <QmeType, T, MatrixType, NumState>)
      .def("init_aux_vars",           &InitAuxVars           <QmeType, T, MatrixType, NumState>)
      // .def("construct_commutator", &construct_commutator       <QmeType, T, MatrixType>)
      // .def("apply_commutator",     &apply_commutator           <QmeType, T, MatrixType>)
      .def("time_evolution",          &TimeEvolution             <QmeType, T, MatrixType, NumState>);
}


template<template <typename,
                   template <typename, int> class,
                   int> class QmeType,
         typename T,
         template<typename, int> class MatrixType, 
         int NumState>
py::class_<QmeType<T, MatrixType, NumState>> declare_qme_gpu_binding(
    py::module m,
    const char* class_name) {
  return py::class_<QmeType<T, MatrixType, NumState>>(m, class_name)
      .def(py::init<>())
      .def("set_Hamiltonian",         &SetHamiltonian        <QmeType, T, MatrixType, NumState>)
      .def("allocate_noises",             &AllocateNoises            <QmeType, T, MatrixType, NumState>)
      .def("set_noise",                   &SetNoise              <QmeType, T, MatrixType, NumState>)
      .def("set_noise_func",    &SetNoiseFunc    <QmeType, T, MatrixType, NumState>)
      .def("init_aux_vars",               &InitAuxVars              <QmeType, T, MatrixType, NumState>)
      // .def("construct_commutator",        &construct_commutator       <QmeType, T, MatrixType>)
      // .def("apply_commutator",            &apply_commutator           <QmeType, T, MatrixType>)
      .def("time_evolution",              &TimeEvolution             <QmeType, T, MatrixType, NumState>)
      .def("set_device_number",           &SetDeviceNumber          <QmeType, T, MatrixType, NumState>);
}


template<template <typename,
                   template <typename, int> class,
                   int> class HeomType,
         typename T,
         template<typename, int> class MatrixType, 
         int NumState>
py::class_<HeomType<T, MatrixType, NumState>> declare_heom_binding(
    py::module m,
    const char* class_name) {
  return py::class_<HeomType<T, MatrixType, NumState>>(m, class_name)
      .def(py::init<>())
      .def("set_Hamiltonian",         &SetHamiltonian        <HeomType, T, MatrixType, NumState>)
      .def("allocate_noises",         &AllocateNoises            <HeomType, T, MatrixType, NumState>)
      .def("set_noise",               &SetNoise              <HeomType, T, MatrixType, NumState>)
      .def("init_aux_vars",           &InitAuxVars              <HeomType, T, MatrixType, NumState>)
      // .def("construct_commutator",        &construct_commutator       <HeomType, T, MatrixType>)
      // .def("apply_commutator",            &apply_commutator           <HeomType, T, MatrixType>)
      .def("time_evolution",              &TimeEvolution             <HeomType, T, MatrixType, NumState>)
      .def("flatten_hierarchy_dimension", &FlattenHierarchyDimension<HeomType, T, MatrixType, NumState>)
      .def("allocate_hierarchy_space",    &AllocateHierarchySpace   <HeomType, T, MatrixType, NumState>);
  // return declare_qme_binding<HeomType, T>(m, class_name)
  //   .def("flatten_hierarchy_dimension", &FlattenHierarchyDimension<HeomType, T>)
  //   .def("allocate_hierarchy_space",    &AllocateHierarchySpace   <HeomType, T>);
}


template<template <typename,
                   template <typename, int> class,
                   int> class HeomType,
         typename T,
         template<typename, int> class MatrixType, 
         int NumState>
py::class_<HeomType<T, MatrixType, NumState>> declare_heom_gpu_binding(
    py::module m,
    const char* class_name) {
  return py::class_<HeomType<T, MatrixType, NumState>>(m, class_name)
      .def(py::init<>())
      .def("set_Hamiltonian",         &SetHamiltonian        <HeomType, T, MatrixType, NumState>)
      .def("allocate_noises",             &AllocateNoises            <HeomType, T, MatrixType, NumState>)
      .def("set_noise",               &SetNoise              <HeomType, T, MatrixType, NumState>)
      .def("init_aux_vars",               &InitAuxVars              <HeomType, T, MatrixType, NumState>)
      // .def("construct_commutator",        &construct_commutator       <HeomType, T, MatrixType>)
      // .def("apply_commutator",            &apply_commutator           <HeomType, T, MatrixType>)
      .def("time_evolution",              &TimeEvolution             <HeomType, T, MatrixType, NumState>)
      .def("flatten_hierarchy_dimension", &FlattenHierarchyDimension<HeomType, T, MatrixType, NumState>)
      .def("allocate_hierarchy_space",    &AllocateHierarchySpace   <HeomType, T, MatrixType, NumState>)
      .def("set_device_number",           &SetDeviceNumber          <HeomType, T, MatrixType, NumState>);
}


PYBIND11_MODULE(pylibheom, m) {
  m.doc() = "Low-level Python Binding of LibHEOM";

  m.attr("support_gpu_parallelization")
      = py::cast(support_gpu_parallelization);
  m.def("gpu_device_count", &GetGpuDeviceCount);
  m.def("gpu_device_name",  &GetGpuDeviceName);

  py::class_<CooMatrix<complex64 >>(m, "CooMatrix_c")
      .def(py::init<int, int, int,
           py::array_t<int>, py::array_t<int>, py::array_t<complex64>>());
  py::class_<CooMatrix<complex128>>(m, "CooMatrix_z")
      .def(py::init<int, int, int,
           py::array_t<int>, py::array_t<int>, py::array_t<complex128>>());
    
  declare_heom_binding<HeomLL,    complex128, DenseMatrix, Eigen::Dynamic>(m, "HEOM_zDLL");
  declare_heom_binding<HeomLL,    complex128, CsrMatrix,   Eigen::Dynamic>(m, "HEOM_zSLL");
  declare_heom_binding<HeomLH,    complex128, DenseMatrix, Eigen::Dynamic>(m, "HEOM_zDLH");
  declare_heom_binding<HeomLH,    complex128, CsrMatrix,   Eigen::Dynamic>(m, "HEOM_zSLH");
  declare_qme_binding <RedfieldH, complex128, DenseMatrix, Eigen::Dynamic>(m, "Redfield_zDH");
  declare_qme_binding <RedfieldH, complex128, CsrMatrix,   Eigen::Dynamic>(m, "Redfield_zSH");
  declare_qme_binding <RedfieldL, complex128, DenseMatrix, Eigen::Dynamic>(m, "Redfield_zDL");
  declare_qme_binding <RedfieldL, complex128, CsrMatrix,   Eigen::Dynamic>(m, "Redfield_zSL");
  // declare_heom_binding<HeomLL,    complex128, DenseMatrix, 2>(m, "HEOM_zDLL");
  // declare_heom_binding<HeomLL,    complex128, CsrMatrix,   2>(m, "HEOM_zSLL");
  // declare_heom_binding<HeomLH,    complex128, DenseMatrix, 2>(m, "HEOM_zDLH");
  // declare_heom_binding<HeomLH,    complex128, CsrMatrix,   2>(m, "HEOM_zSLH");
  // declare_qme_binding <RedfieldH, complex128, DenseMatrix, 2>(m, "Redfield_zDH");
  // declare_qme_binding <RedfieldH, complex128, CsrMatrix,   2>(m, "Redfield_zSH");
  // declare_qme_binding <RedfieldL, complex128, DenseMatrix, 2>(m, "Redfield_zDL");
  // declare_qme_binding <RedfieldL, complex128, CsrMatrix,   2>(m, "Redfield_zSL");
#ifdef SUPPORT_GPU_PARALLELIZATION
  declare_heom_gpu_binding<HeomLHGpu,   complex128, DenseMatrix, Eigen::Dynamic>(m, "HEOM_zDLH_GPU");
  declare_heom_gpu_binding<HeomLHGpu,   complex128, CsrMatrix,   Eigen::Dynamic>(m, "HEOM_zSLH_GPU");
  declare_qme_gpu_binding <RedfieldHGpu,complex128, DenseMatrix, Eigen::Dynamic>(m, "Redfield_zDH_GPU");
  declare_qme_gpu_binding <RedfieldHGpu,complex128, CsrMatrix,   Eigen::Dynamic>(m, "Redfield_zSH_GPU");
  // declare_qme_gpu_binding <RedfieldLGpu,complex128, DenseMatrix, Eigen::Dynamic>(m, "Redfield_zDL_GPU");
  // declare_qme_gpu_binding <RedfieldLGpu,complex128, CsrMatrix,   Eigen::Dynamic>(m, "Redfield_zSL_GPU");
#endif
}
