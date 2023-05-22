/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef LIBHEOM_UTILITY_MKL_H
#define LIBHEOM_UTILITY_MKL_H

#ifdef ENABLE_MKL
#include "linalg_engine/include_mkl.h"
#include <iostream>
#include <map>
#include <string>

namespace libheom {

extern std::map<sparse_status_t, std::string> MKL_SPARSE_ERR_MSG;
#define MKL_SPARSE_CALL(func)                                                    \
        {                                                                        \
          sparse_status_t err = (func);                                          \
          if (err != SPARSE_STATUS_SUCCESS) {                                    \
            std::cerr << "[Error:mkl]  "                                         \
                      << "(error code: " << err << ") "                          \
                      << "at " << __FILE__ << " line " << __LINE__ << std::endl; \
            std::cerr << MKL_SPARSE_ERR_MSG[err] << std::endl;                   \
            std::exit(1);                                                        \
          }                                                                      \
        }

}
#endif // ifdef ENABLE_MKL
#endif // ifndef LIBHEOM_UTILITY_MKL_H
