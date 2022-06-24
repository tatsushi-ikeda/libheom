/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#include "utility.h"

#include <omp.h>

namespace libheom {

std::vector<std::string> stack_funcname;

stack_funcname_operation::stack_funcname_operation(std::string str)
{
  if (omp_get_thread_num() == 0) {
    stack_funcname.push_back(str);
  }
}

stack_funcname_operation::~stack_funcname_operation()
{
  if (omp_get_thread_num() == 0) {
    stack_funcname.pop_back();
  }
}

void terminate_handler()
{
#ifdef QL_NDEBUG
  std::cerr << "Call trace is deactivated. Compile without -DQL_NDEBUG." << std::endl;
#else
  int i = 0;
  std::cerr << "============ Call Trace ============" << std::endl;
  for (const auto& elem : stack_funcname ) {
    std::cerr << "  " << ++i << " : " << elem << std::endl;
  }
  std::cerr << "====================================" << std::endl;
#endif
}

void sigsegv_handler(int nSignum, siginfo_t* si, void* vcontext)
{
  terminate_handler();
  std::cerr << "segmentation fault" << std::endl;
  std::exit(127);
}

};
