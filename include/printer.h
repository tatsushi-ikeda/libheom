/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef PRINTER_H
#define PRINTER_H

#include <tuple>
#include <vector>
#include <iostream>
#include <fstream>

namespace libheom {

class ShapePrinter {
public:
  ShapePrinter(const std::tuple<int, int>& shape) :
    data(shape)
  {}
  const std::tuple<int, int>& data;
};

std::ostream& operator <<
/**/(std::ostream& out,
     const ShapePrinter& printer);


template <typename T>
class VectorPrinter {
public:
  VectorPrinter(const std::vector<T>& vec) :
    data(vec)
  {}
  const std::vector<T>& data;
};

template <typename T>
std::ostream& operator <<
/**/(std::ostream& out,
     const VectorPrinter<T>& printer)
{
  std::size_t size = printer.data.size();
  out << "[";
  for (std::size_t i = 0; i < size-1; ++i) {
    out << printer.data[i] << ", " ;
  }
  out << printer.data[size-1];
  out << "]";
  return out;
}

}


#endif
