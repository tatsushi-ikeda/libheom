/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LICENSE.txt for license.
 *------------------------------------------------------------------------*/

#include "hierarchy_space.h"

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
  for (dtype i = 1; i <= r; ++i) {
    num *= n + i - 1;
    den *= i;
    dtype gcd = calc_gcd(num, den);
    num /= gcd;
    den /= gcd;
  }
  return num/den;
}

long long calc_hierarchy_element_count(int level, int dim)
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

int alloc_hierarchy_space(hierarchy_space& hs,
                      int  truncation_depth,
                      std::function<void(int, int)> callback,
                      int  interval_callback,
                      std::function<bool(vector<int>, int)> hierarchy_filter,
                      bool filter_flag)
{
  int n_modes = hs.n_modes;

  vector<int> index(n_modes);
  int lidx = 0;
  std::queue<vector<int>> next_element;
  std::queue<int>         k_last_modified;

  long long estimated_max_lidx = calc_hierarchy_element_count(truncation_depth, n_modes);
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

    hs.multi_index_map[index] = lidx;
    hs.n.push_back(index);
    ++lidx;

    for (int k = last_modified; k < n_modes; ++k) {
      ++index[k];
      int depth = std::accumulate(index.begin(), index.end(), 0);
      bool pass = (depth <= truncation_depth) && (!filter_flag || hierarchy_filter(index, depth));
      if (pass) {
        if (depth == truncation_depth && filter_flag) {
          std::cerr << "[Warning]: hierarchy_filter has reached truncation_depth ";
          print_index(index, std::cerr);
          std::cerr << std::endl;
        }
        next_element.push(index);
        k_last_modified.push(k);
      }
      --index[k];
    }
  }

  int n_hierarchy = lidx;
  hs.ptr_void = lidx;

  // Look up the linear index of a neighbor; return ptr_void if out of range.
  auto look_up = [&](const vector<int>& idx) -> int {
    auto it = hs.multi_index_map.find(idx);
    return (it != hs.multi_index_map.end()) ? it->second : hs.ptr_void;
  };

  hs.ptr_p1.resize(n_hierarchy + 1);
  hs.ptr_m1.resize(n_hierarchy + 1);
  for (int i = 0; i < n_hierarchy; ++i) {
    index = hs.n[i];
    hs.ptr_p1[i].resize(n_modes);
    hs.ptr_m1[i].resize(n_modes);
    for (int k = 0; k < n_modes; ++k) {
      ++index[k];
      hs.ptr_p1[i][k] = look_up(index);
      index[k] -= 2;
      hs.ptr_m1[i][k] = look_up(index);
      ++index[k];
    }
  }
  hs.ptr_p1[hs.ptr_void].assign(n_modes, hs.ptr_void);
  hs.ptr_m1[hs.ptr_void].assign(n_modes, hs.ptr_void);
  return n_hierarchy;
}

}
