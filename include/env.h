/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef ENV_H
#define ENV_H

#include "type.h"
#include "utility.h"

namespace libheom {

class env_base
{
};

template <typename dtype, typename env>
struct device_type;

template<typename dtype, typename env>
using device_t = typename device_type<dtype,env>::value;

template <typename dtype, typename env, bool mirror> struct new_dev_impl;
template <typename dtype, typename env, bool mirror=false>
device_t<dtype,env>* new_dev (int size)
{
  CALL_TRACE();
  return new_dev_impl<dtype,env,mirror>::func(size);
}


template <typename dtype, typename env, bool mirror> struct delete_dev_impl;
template <typename dtype, typename env, bool mirror=false>
void delete_dev(device_t<dtype,env>* ptr)
{
  CALL_TRACE();
  delete_dev_impl<dtype,env,mirror>::func(ptr);
}


template <typename dtype, typename env> struct host2dev_impl;
template <typename dtype, typename env>
void host2dev(dtype* const & ptr_host, device_t<dtype,env>*& ptr_dev, int size)
{
  CALL_TRACE();
  return host2dev_impl<dtype,env>::func(ptr_host, ptr_dev, size);
}


template <typename dtype, typename env> struct dev2host_impl;
template <typename dtype, typename env>
void dev2host(device_t<dtype,env>*& ptr_dev, dtype* const & ptr_host, int size)
{
  CALL_TRACE();
  return dev2host_impl<dtype,env>::func(ptr_dev, ptr_host, size);
}


}

#include <new>
#ifdef __INTEL_COMPILER
#include <aligned_new>
#endif

namespace libheom {
class env_cpu: public env_base
{
};

template <> struct device_type<int,       env_cpu> { typedef int value; };
template <> struct device_type<float32,   env_cpu> { typedef float value; };
template <> struct device_type<float64,   env_cpu> { typedef double value; };
template <> struct device_type<complex64, env_cpu> { typedef complex64 value; };
template <> struct device_type<complex128,env_cpu> { typedef complex128 value; };


template <>
struct new_dev_impl<complex64,env_cpu,false>
{
  inline static device_t<complex64,env_cpu>* func(int size)
  {
    CALL_TRACE();
    return new (std::align_val_t{align_val<complex64>}) device_t<complex64,env_cpu> [size];  //
  }
};

template <>
struct new_dev_impl<complex128,env_cpu,false>
{
  inline static device_t<complex128,env_cpu>* func(int size)
  {
    CALL_TRACE();
    return new (std::align_val_t{align_val<complex128>}) device_t<complex128,env_cpu> [size];  //
  }
};

template <typename dtype>
struct new_dev_impl<dtype,env_cpu,true>
{
  inline static device_t<dtype,env_cpu>* func(int size)
  {
    CALL_TRACE();
    return nullptr;
  }
};


template <typename dtype>
struct delete_dev_impl<dtype,env_cpu,false>
{
  inline static void func(device_t<dtype,env_cpu>* ptr)
  {
    CALL_TRACE();
    delete [] ptr;
  }
};

template <typename dtype>
struct delete_dev_impl<dtype,env_cpu,true>
{
  inline static void func(device_t<dtype,env_cpu>* ptr)
  {
    CALL_TRACE();
  }
};


template <typename dtype>
struct host2dev_impl<dtype,env_cpu>
{
  inline static void func(dtype* const & ptr_host, device_t<dtype,env_cpu>*& ptr_dev, int size)
  {
    CALL_TRACE();
    ptr_dev = ptr_host;
  }
};


template <typename dtype>
struct dev2host_impl<dtype,env_cpu>
{
  inline static void func(device_t<dtype,env_cpu>*& ptr_dev, dtype* const & ptr_host, int size)
  {
    CALL_TRACE();
    // ptr_host = ptr_dev;
  }
};


}

#endif
