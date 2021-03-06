/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef TYPE_GPU_H
#define TYPE_GPU_H

#include <cublas_v2.h>
#include <cusparse.h>
#include <thrust/complex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "type.h"
#include "dense_matrix.h"
#include "csr_matrix.h"
// #include "dense_matrix_gpu.h"
// #include "csr_matrix_gpu.h"

namespace libheom {

// Get the corresponding type name on thrust library from type name on cpu code.
#define GPU_TYPE(type) typename GpuType<type>::value
template <typename T> struct GpuType;
template<> struct GpuType<int>        { typedef int value; };
template<> struct GpuType<float32>    { typedef float value; };
template<> struct GpuType<float64>    { typedef double value; };
template<> struct GpuType<complex64>  { typedef thrust::complex<float> value; };
template<> struct GpuType<complex128> { typedef thrust::complex<double> value; };

// Get the corresponding type name on device code from type name on cpu code.
template <typename T> struct RawGpuType;
#define RAW_GPU_TYPE(type) typename RawGpuType<type>::value
#define DECLARE_RAW_GPU_TYPE(type, GpuType) \
  template<> struct RawGpuType<type>        { typedef GpuType value; }; \
  template<> struct RawGpuType<const type>  { typedef const GpuType value; }; \
  template<> struct RawGpuType<type*>       { typedef GpuType* value; }; \
  template<> struct RawGpuType<const type*> { typedef const GpuType* value; };
DECLARE_RAW_GPU_TYPE(int,        int);
DECLARE_RAW_GPU_TYPE(float32,     float);
DECLARE_RAW_GPU_TYPE(float64,     double);
DECLARE_RAW_GPU_TYPE(complex64,  cuFloatComplex);
DECLARE_RAW_GPU_TYPE(complex128, cuDoubleComplex);
#undef DECLARE_RAW_GPU_TYPE


// Implemention of raw_gpu_type_cast<>() below.
template<typename T>
struct RawGpuTypeCastImpl {
  template<typename U>
  static inline RAW_GPU_TYPE(T) cast (U data);
};
template<typename T> template<typename U>
inline RAW_GPU_TYPE(T) RawGpuTypeCastImpl<T>::cast(U data) {
  return reinterpret_cast<RAW_GPU_TYPE(T)>(thrust::raw_pointer_cast(data));
}
template<> template<>
inline RAW_GPU_TYPE(complex64) RawGpuTypeCastImpl<complex64>::cast<float32>(float32 data) {
  return make_cuFloatComplex(data, 0.0f);
}
template<> template<>
inline RAW_GPU_TYPE(complex64) RawGpuTypeCastImpl<complex64>::cast<complex64>(complex64 data) {
  return make_cuFloatComplex(std::real(data), std::imag(data));
}
template<> template<>
inline RAW_GPU_TYPE(complex128) RawGpuTypeCastImpl<complex128>::cast<float64>(float64 data) {
  return make_cuDoubleComplex(data, 0.0);
}
template<> template<>
inline RAW_GPU_TYPE(complex128) RawGpuTypeCastImpl<complex128>::cast<complex128>(complex128 data) {
  return make_cuDoubleComplex(std::real(data), std::imag(data));
}
// Type casting from cpu type or thrust type to device type.
// Note: To imitate the standard cast notation, snake case name is used.
template<typename T, typename U>
inline RAW_GPU_TYPE(T) raw_gpu_type_cast(U data) {
  return RawGpuTypeCastImpl<T>::cast(data);
}


template<typename T>
inline void CopyVectorGpu(const std::vector<T>& input,
                          thrust::device_vector<GPU_TYPE(T)>& output) {
  output.resize(input.size());
  thrust::copy_n(reinterpret_cast<const GPU_TYPE(T)*>(input.data()),
                 input.size(), output.begin());
}
template<typename T>
inline void CopyVectorGpu(const thrust::device_vector<GPU_TYPE(T)>& input,
                          std::vector<T>& output) {
  output.resize(input.size());
  thrust::copy_n(input.begin(), input.size(),
                 reinterpret_cast<GPU_TYPE(T)*>(output.data()));
}
template<typename T>
inline void CopyVectorGpu(const T* input,
                          thrust::device_vector<GPU_TYPE(T)>& output) {
  thrust::copy_n(reinterpret_cast<const GPU_TYPE(T)*>(input),
                 output.size(), output.begin());
}
template<typename T>
inline void CopyVectorGpu(const thrust::device_vector<GPU_TYPE(T)>& input,
                          T* output) {
  thrust::copy_n(input.begin(), input.size(),
                 reinterpret_cast<GPU_TYPE(T)*>(output));
}


template <typename T> class DenseMatrixGpu;
template <typename T>
class CsrMatrixGpu {
 public:
  std::tuple<int, int> shape;
  int nnz;
  
  thrust::device_vector<GPU_TYPE(T)> data;
  thrust::device_vector<int> indices;
  thrust::device_vector<int> indptr;

  inline CsrMatrixGpu<T>& operator = (const CsrMatrix<T, Eigen::Dynamic>& rhs) {
    std::get<0>(this->shape) = rhs.rows();
    std::get<1>(this->shape) = rhs.cols();
    
    this->nnz     = rhs.nonZeros();
    
    // this->data = rhs.data;
    // Note: On Windows system, a simple substitution from
    //       std::vector<std::complex<T>> to
    //       thrust::device_vector<thrust::complex<T>> causes error.
    this->data.resize(this->nnz);
    this->indices.resize(this->nnz);
    this->indptr.resize(std::get<0>(this->shape) + 1);
    
    CopyVectorGpu(rhs.valuePtr(),      this->data);
    CopyVectorGpu(rhs.innerIndexPtr(), this->indices);
    CopyVectorGpu(rhs.outerIndexPtr(), this->indptr);

    // std::ofstream out("tmp.dat");
    // thrust::host_vector<int> tmp;
    
    // out << "indices" << std::endl;
    // tmp = this->indices;
    // for (int i = 0; i < tmp.size(); ++i) {
    //   out << tmp[i] << std::endl;
    // }

    // out << "indptr" << std::endl;
    // tmp = this->indptr;
    // for (int i = 0; i < tmp.size(); ++i) {
    //   out << tmp[i] << std::endl;
    // }
    
    // thrust::host_vector<T> tmp2;
    // out << "value" << std::endl;
    // tmp2 = this->data;
    // for (int i = 0; i < tmp2.size(); ++i) {
    //   out << tmp2[i] << std::endl;
    // }
    
    return (*this);
  };
};


// Get the corresponding matrix type name on device from matrix name on cpu code.
#define GPU_MATRIX_TYPE(type) typename GpuMatrixType<type>::value
template <template <typename, int> class T> struct GpuMatrixType;
template<> struct GpuMatrixType<DenseMatrix> {
  template <typename T>
  using value = DenseMatrixGpu<T>;
};
template<> struct GpuMatrixType<CsrMatrix> {
  template <typename T>
  using value = CsrMatrixGpu<T>;
};

}
#endif /* TYPE_GPU_H */
