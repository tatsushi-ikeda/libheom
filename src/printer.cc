/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#include "printer.h"

namespace libheom {

std::ostream& operator << (std::ostream& out,
                           const ShapePrinter& printer) {
  out << "("
      << std::get<0>(printer.data) << ", "
      << std::get<1>(printer.data) << ")";
  return out;
}

}
