/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <vector>
#include <signal.h>

#if (defined(__ICC) && !defined(__NVCC__))
#define INLINE __forceinline
#else
#define INLINE inline
#endif

#ifdef QL_NDEBUG
#  define CALL_TRACE() {}
#else
#  define CALL_TRACE() stack_funcname_operation call_trace_temp_symbol(std::string(__func__) + " @ " + std::string(__FILE__) + ", " + std::to_string(__LINE__));
#endif

namespace libheom {

class stack_funcname_operation {
 public:
  stack_funcname_operation(std::string str);
  ~stack_funcname_operation();
};

#if !defined(QL_NDEBUG)
extern std::vector<std::string> stack_funcname;
#endif
void terminate_handler();
void sigsegv_handler(int nSignum, siginfo_t* si, void* vcontext);

};

#endif
