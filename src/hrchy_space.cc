/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LICENSE.txt for licence.
 *------------------------------------------------------------------------*/

#include "hrchy_space.h"

#include <queue>
#include <numeric>
#include <stdexcept>
#include <iostream>

namespace libheom
{

template<typename dtype>
dtype calc_gcd(dtype m, dtype n)
{
  if (m < n) {
    std::swap(m, n);
  }
  while (n != 0) {
    dtype swap = n;
    n = m%n;
    m = swap;
  }
  return m;
}


template<typename dtype>
dtype calc_multicombination(dtype n, dtype r)
{
  dtype num, den;
  num = 1;
  den = 1;
  for(dtype i = 1; i <= r; ++i) {
    num *= n + i - 1;
    den *= i;
    dtype gcd = calc_gcd(num, den);
    num /= gcd;
    den /= gcd;
  }
  return num/den;
}


long long calc_hrchy_element_count(int level, int dim)
{
  return calc_multicombination<long long>(dim + 1, level);
}


void print_index(vector<int>& index, std::ostream& out)
{
  out << "[";
  if (index.size() > 0) {
    out << index[0];
  }
  for (int k = 1; k < index.size(); ++k) {
    out << ", " << index[k];
  }
  out << "]";
}


int alloc_hrchy_space(hrchy_space& hs,
                      int  max_depth,
                      std::function<void(int, int)> callback,
                      int  interval_callback,
                      std::function<bool(vector<int>, int)> hrchy_filter,
                      bool filter_flag)
{
  int n_dim = hs.n_dim;

  vector<int> index(n_dim);
  int lidx = 0;
  std::queue<vector<int>> next_element;
  std::queue<int>         k_last_modified;

  long long estimated_max_lidx = calc_hrchy_element_count(max_depth, n_dim);
  std::fill(index.begin(), index.end(), 0);
  next_element.push(index);
  k_last_modified.push(0);

  while (!next_element.empty()) {
    if (lidx % interval_callback == 0) {
      callback(lidx, estimated_max_lidx);
    }
    index = next_element.front();
    next_element.pop();
    int last_modified = k_last_modified.front();
    k_last_modified.pop();

    hs.book[index] = lidx;
    hs.n.push_back(index);
    ++lidx;

    for (int k = last_modified; k < n_dim; ++k) {
      ++index[k];
      int depth = std::accumulate(index.begin(), index.end(), 0);
      bool pass = (depth <= max_depth) && (!filter_flag || hrchy_filter(index, depth));
      if (pass) {
        if (depth == max_depth && filter_flag) {
          std::cerr << "[Warning]: hrchy_filter has reached max_depth ";
          print_index(index, std::cerr);
          std::cerr << std::endl;
        }
        next_element.push(index);
        k_last_modified.push(k);
      }
      --index[k];
    }
  }

  int n_hrchy = lidx;
  hs.ptr_void = lidx;

  // Look up the linear index of a neighbor; return ptr_void if out of range.
  auto look_up = [&](const vector<int>& idx) -> int {
    auto it = hs.book.find(idx);
    return (it != hs.book.end()) ? it->second : hs.ptr_void;
  };

  hs.ptr_p1.resize(n_hrchy + 1);
  hs.ptr_m1.resize(n_hrchy + 1);
  for (int i = 0; i < n_hrchy; ++i) {
    index = hs.n[i];
    hs.ptr_p1[i].resize(n_dim);
    hs.ptr_m1[i].resize(n_dim);
    for (int k = 0; k < n_dim; ++k) {
      ++index[k];
      hs.ptr_p1[i][k] = look_up(index);
      index[k] -= 2;
      hs.ptr_m1[i][k] = look_up(index);
      ++index[k];
    }
  }
  hs.ptr_p1[hs.ptr_void].assign(n_dim, hs.ptr_void);
  hs.ptr_m1[hs.ptr_void].assign(n_dim, hs.ptr_void);
  return n_hrchy;
}

}
