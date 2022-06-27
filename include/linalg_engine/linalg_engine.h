/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef LINALG_ENGINE_H
#define LINALG_ENGINE_H

#include <stdexcept>
#include "type.h"
#include "env.h"
#include "env_gpu.h"
#include "utility.h"

#include "linalg_engine/matrix_base.h"
#include "linalg_engine/lil_matrix.h"

namespace libheom
{

constexpr int dynamic = 0;

class linalg_engine_base
{
 public:
  
  int n_children;
  linalg_engine_base** children;
  int n_threads_old;
  
  linalg_engine_base() : children(nullptr), n_children(0), n_threads_old(0)
  {
    CALL_TRACE();
  }

  linalg_engine_base(linalg_engine_base* parent) : children(nullptr), n_children(0), n_threads_old(0)
  {
    CALL_TRACE();
  }

  virtual ~linalg_engine_base()
  {
    CALL_TRACE();
    destroy_children();
  }

  virtual linalg_engine_base* create_child() = 0;
  
  void create_children(int n)
  {
    CALL_TRACE();
    if (n > this->n_children) {
      destroy_children();
      this->n_children = n;
      this->children = new linalg_engine_base*[n];
      for (int i = 0; i < n; ++i) {
        this->children[i] = create_child ();
      }
    }
  }
  
  void destroy_children()
  {
    CALL_TRACE();
    if (this->n_children > 0) {
      for (int i = 0; i < this->n_children; ++i) {
        delete this->children[i];
      }
      delete [] this->children;
      this->children = nullptr;
      this->n_children = 0;
    }
  }

  linalg_engine_base* get_child(int i)
  {
    return this->children[i];
  }

  virtual void set_n_inner_threads(int n)
  {
    CALL_TRACE();
  }

  virtual void set_n_outer_threads(int n)
  {
    CALL_TRACE();
  }

  virtual void switch_thread(int n)
  {
    CALL_TRACE();
  }
};

using not_implemented = linalg_engine_base;

template <typename linalg_engine>
struct engine_env_impl;

template <typename linalg_engine>
using engine_env = typename engine_env_impl<linalg_engine>::value;

template <typename linalg_engine>
INLINE bool is_supported() { return false; }

using nil = void;
// class nil : public linalg_engine_base   {};
template <> struct engine_env_impl<nil> { typedef env_cpu value; };
template <> INLINE bool is_supported<nil>() { return true; }

constexpr nil* nilobj = nullptr;


class eigen;
#ifdef ENABLE_EIGEN
template <> struct engine_env_impl<eigen> { typedef env_cpu value; };
template <> INLINE bool is_supported<eigen>()    { return true; }
#endif

class mkl;
#ifdef ENABLE_MKL
template <> struct engine_env_impl<mkl>   { typedef env_cpu value; };
template <> INLINE bool is_supported<mkl>()      { return true; }
#else
#endif

class cuda;
#ifdef ENABLE_CUDA
template <> struct engine_env_impl<cuda>  { typedef env_gpu value; };
template <> INLINE bool is_supported<cuda>()     { return true; }
#endif

template<int n_level_c, typename dtype_dev, typename linalg_engine = not_implemented>
struct nullify_impl;

template<int n_level_c, typename dtype_dev, typename linalg_engine>
INLINE void nullify(linalg_engine* engine_obj,
                    dtype_dev* x,
                    int n_level)
{
  CALL_TRACE();
  nullify_impl<n_level_c,dtype_dev,linalg_engine>::func(engine_obj, x, n_level);
}


template<int n_level_c, typename dtype_dev, typename linalg_engine = not_implemented>
struct copy_impl;

template<int n_level_c, typename dtype_dev, typename linalg_engine>
INLINE void copy(linalg_engine* engine_obj,
                 dtype_dev* x,
                 dtype_dev* y,
                 int n_level)
{
  CALL_TRACE();
  copy_impl<n_level_c,dtype_dev,linalg_engine>::func(engine_obj, x, y, n_level);
}


template<int n_level_c, typename dtype, typename linalg_engine = not_implemented>
struct scal_impl;

template< int n_level_c, typename dtype, typename linalg_engine>
INLINE void scal(linalg_engine* engine_obj,
                 dtype a,
                 device_t<dtype,engine_env<linalg_engine>>* y,
                 int n_level)
{
  CALL_TRACE();
  scal_impl<n_level_c,dtype,linalg_engine>::func(engine_obj, a, y, n_level);
}


template<int n_level_c, typename dtype, typename linalg_engine = not_implemented>
struct axpy_impl;

template<int n_level_c, typename dtype, typename linalg_engine>
INLINE void axpy(linalg_engine* engine_obj,
                       dtype a,
                       device_t<dtype,engine_env<linalg_engine>>* x,
                       device_t<dtype,engine_env<linalg_engine>>* y,
                       int n_level)
{
  CALL_TRACE();
  axpy_impl<n_level_c,dtype,linalg_engine>::func(engine_obj, a, x, y, n_level);
}


template<int n_level_c, typename dtype_dev, typename linalg_engine = not_implemented>
struct dotc_impl;

template<int n_level_c, typename dtype_dev, typename linalg_engine>
INLINE dtype_dev dotc(linalg_engine* engine_obj,
                      dtype_dev* x,
                      dtype_dev* y,
                      int n_level)
{
  CALL_TRACE();
  return dotc_impl<n_level_c,dtype_dev,linalg_engine>::func(engine_obj, x, y, n_level);
}


template<int n_level_c, typename dtype_dev, typename linalg_engine = not_implemented>
struct dotu_impl;

template<int n_level_c, typename dtype_dev, typename linalg_engine>
INLINE dtype_dev dotu(linalg_engine* engine_obj,
                      dtype_dev* x,
                      dtype_dev* y,
                      int n_level)
{
  CALL_TRACE();
  return dotu_impl<n_level_c,dtype_dev,linalg_engine>::func(engine_obj, x, y, n_level);
}


template<int n_level_c, typename dtype_real, typename linalg_engine = not_implemented>
struct errnrm1_impl;

template<int n_level_c, typename dtype_real, typename linalg_engine>
INLINE dtype_real errnrm1(linalg_engine* engine_obj,
                          device_t<complex_t<dtype_real>,engine_env<linalg_engine>>* e,
                          device_t<complex_t<dtype_real>,engine_env<linalg_engine>>* x_1,
                          device_t<complex_t<dtype_real>,engine_env<linalg_engine>>* x_2,
                          dtype_real atol,
                          dtype_real rtol,
                          int n_level)
{
  CALL_TRACE();
  return errnrm1_impl<n_level_c,dtype_real,linalg_engine>::func(engine_obj, e, x_1, x_2, atol, rtol, n_level);
}

template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   order_t order_,
                   typename linalg_engine_> class matrix_base,
         order_t order,
         typename linalg_engine = not_implemented>
struct axpym_impl;

template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   order_t order_,
                   typename linalg_engine_> class matrix_base,
         order_t order,
         typename linalg_engine = not_implemented>
INLINE void axpy(linalg_engine* engine_obj,
                 dtype a,
                 matrix_base<n_level_c,dtype,order,linalg_engine>& x,
                 matrix_base<n_level_c,dtype,order,linalg_engine>& y,
                 int n_level)
{
  CALL_TRACE();
  axpym_impl<n_level_c,dtype,matrix_base,order,linalg_engine>::func(engine_obj, a, x, y, n_level);
}

template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   order_t order_,
                   typename linalg_engine_> class matrix_base,
         order_t order,
         typename linalg_engine = not_implemented>
struct gemm_impl;

template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   order_t order_,
                   typename linalg_engine_> class matrix_base,
         order_t order,
         typename linalg_engine>
INLINE void gemm(linalg_engine* engine_obj,
                 dtype alpha,
                 matrix_base<n_level_c,dtype,order,linalg_engine>& A,
                 device_t<dtype,engine_env<linalg_engine>>* B,
                 dtype beta,
                 device_t<dtype,engine_env<linalg_engine>>* C,
                 int n_level)
{
  CALL_TRACE();
  gemm_impl<n_level_c,dtype,matrix_base,order,linalg_engine>::func(engine_obj, alpha, A, B, beta, C, n_level);
}


template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   order_t order_,
                   typename linalg_engine_> class matrix_base,
         order_t order,
         typename linalg_engine>
INLINE void gemm(linalg_engine* engine_obj,
                 dtype alpha,
                 device_t<dtype,engine_env<linalg_engine>>* A,
                 matrix_base<n_level_c,dtype,order,linalg_engine>& B,
                 dtype beta,
                 device_t<dtype,engine_env<linalg_engine>>* C,
                 int n_level)
{
  CALL_TRACE();
  gemm_impl<n_level_c,dtype,matrix_base,order,linalg_engine>::func(engine_obj, alpha, A, B, beta, C, n_level);
}

template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   order_t order_,
                   typename linalg_engine_> class matrix_base,
         order_t order,
         typename linalg_engine>
INLINE void gemm(linalg_engine* engine_obj,
                 dtype alpha,
                 matrix_base<n_level_c,dtype,order,linalg_engine>& A,
                 matrix_base<n_level_c,dtype,order,linalg_engine>& B,
                 dtype beta,
                 matrix_base<n_level_c,dtype,order,linalg_engine>& C,
                 int n_level)
{
  CALL_TRACE();
  gemm_impl<n_level_c,dtype,matrix_base,order,linalg_engine>::func(engine_obj, alpha, A, B, beta, C, n_level);
}

template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   order_t order_,
                   typename linalg_engine_> class matrix_base,
         order_t order,
         typename linalg_engine = not_implemented>
struct gemv_impl;

template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   order_t order_,
                   typename linalg_engine_> class matrix_base,
         order_t order,
         typename linalg_engine>
INLINE void gemv(linalg_engine* engine_obj,
                 dtype alpha,
                 matrix_base<n_level_c,dtype,order,linalg_engine>& A,
                 device_t<dtype,engine_env<linalg_engine>>* x,
                 dtype beta,
                 device_t<dtype,engine_env<linalg_engine>>* y,
                 int n_level)
{
  CALL_TRACE();
  gemv_impl<n_level_c,dtype,matrix_base,order,linalg_engine>::func(engine_obj, alpha, A, x, beta, y, n_level);
}


template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   order_t order_,
                   typename linalg_engine_> class matrix_base,
         order_t order,
         typename linalg_engine = not_implemented>
struct gevm_impl;

template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   order_t order_,
                   typename linalg_engine_> class matrix_base,
         order_t order,
         typename linalg_engine>
INLINE void gevm(linalg_engine* engine_obj,
                 dtype alpha,
                 device_t<dtype,engine_env<linalg_engine>>* x,
                 matrix_base<n_level_c,dtype,order,linalg_engine>& A,
                 dtype beta,
                 device_t<dtype,engine_env<linalg_engine>>* y,
                 int n_level)
{
  CALL_TRACE();
  gevm_impl<n_level_c,dtype,matrix_base,order,linalg_engine>::func(engine_obj, alpha, x, A, beta, y, n_level);
}


template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   order_t order_,
                   typename linalg_engine_> class matrix_base,
         order_t order,
         typename linalg_engine = not_implemented>
struct eig_impl;

template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   order_t order_,
                   typename linalg_engine_> class matrix_base,
         order_t order,
         typename linalg_engine>
INLINE void eig(linalg_engine* engine_obj,
                matrix_base<n_level_c,dtype,order,linalg_engine>& A,
                device_t<real_t<dtype>,engine_env<linalg_engine>>* w,
                device_t<dtype,engine_env<linalg_engine>>* v,
                int n_level)
{
  CALL_TRACE();
  eig_impl<n_level_c,dtype,matrix_base,order,linalg_engine>::func(engine_obj, A, w, v, n_level);
}


template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   order_t order_,
                   typename linalg_engine_> class matrix_base,
         order_t order,
         typename linalg_engine = not_implemented>
struct utf_impl;


template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   order_t order_,
                   typename linalg_engine_> class matrix_base,
         order_t order,
         typename linalg_engine>
INLINE void utf(linalg_engine* engine_obj,
                matrix_base<n_level_c,dtype,order,linalg_engine>& A,
                device_t<dtype,engine_env<linalg_engine>>* v,
                device_t<dtype,engine_env<linalg_engine>>* A_v,
                int n_level)
{
  CALL_TRACE();
  utf_impl<n_level_c,dtype,matrix_base,order,linalg_engine>::func(engine_obj, A, v, A_v, n_level);
}


template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   order_t order_,
                   typename linalg_engine_> class matrix_base,
         order_t order,
         typename linalg_engine = not_implemented>
struct utb_impl;


template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   order_t order_,
                   typename linalg_engine_> class matrix_base,
         order_t order,
         typename linalg_engine>
INLINE void utb(linalg_engine* engine_obj,
                device_t<dtype,engine_env<linalg_engine>>* A_v,
                device_t<dtype,engine_env<linalg_engine>>* v,
                matrix_base<n_level_c,dtype,order,linalg_engine>& A,
                int n_level)
{
  CALL_TRACE();
  utb_impl<n_level_c,dtype,matrix_base,order,linalg_engine>::func(engine_obj, A_v, v, A, n_level);
}


template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   order_t order_,
                   typename linalg_engine_> class matrix_base,
         order_t order,
         order_t order_kron,
         typename linalg_engine>
struct kron_x_1_impl
{
  INLINE static void func(linalg_engine* obj,
                          const dtype& alpha,
                          const matrix_base<n_level_c,dtype,order,linalg_engine>& x,
                          const dtype& beta,
                          matrix_base<n_level_c,dtype,order_kron,linalg_engine>& y,
                          bool conj=false);
};

template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   order_t order_,
                   typename linalg_engine_> class matrix_base,
         order_t order,
         order_t order_kron,
         typename linalg_engine>
INLINE void kron_x_1(linalg_engine* obj,
                     const dtype& alpha,
                     const matrix_base<n_level_c,dtype,order,linalg_engine>& x,
                     const dtype& beta,
                     matrix_base<n_level_c,dtype,order_kron,linalg_engine>& y,
                     bool conj=false)
{
  CALL_TRACE();
  kron_x_1_impl<n_level_c,dtype,matrix_base,order,order_kron,linalg_engine>::func(obj, alpha, x, beta, y, conj);
}

template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   order_t order_,
                   typename linalg_engine_> class matrix_base,
         order_t order,
         order_t order_kron,
         typename linalg_engine>
struct kron_1_x_T_impl
{
  INLINE static void func(linalg_engine* obj,
                          const dtype& alpha,
                          const matrix_base<n_level_c,dtype,order,linalg_engine>& x,
                          const dtype& beta,
                          matrix_base<n_level_c,dtype,order_kron,linalg_engine>& y,
                          bool conj=false);
};

template<int n_level_c,
         typename dtype,
         template <int n_level_c_,
                   typename dtype_,
                   order_t order_,
                   typename linalg_engine_> class matrix_base,
         order_t order,
         order_t order_kron,
         typename linalg_engine>
INLINE void kron_1_x_T(linalg_engine* obj,
                       const dtype& alpha,
                       const matrix_base<n_level_c,dtype,order,linalg_engine>& x,
                       const dtype& beta,
                       matrix_base<n_level_c,dtype,order_kron,linalg_engine>& y,
                       bool conj=false)
{
  CALL_TRACE();
  kron_1_x_T_impl<n_level_c,dtype,matrix_base,order,order_kron,linalg_engine>::func(obj,alpha, x, beta, y, conj);
}


}

#endif
