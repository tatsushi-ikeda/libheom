/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#include "hierarchy_space.h"

#define ORIGINAL_ORDER      0
#define BREADTH_FIRST_ORDER 1
#define DEPTH_FIRST_ORDER   2
#define ORDER_TYPE  BREADTH_FIRST_ORDER

#if   ORDER_TYPE == BREADTH_FIRST_ORDER
#  include <queue>
#  define AUX_STRUCT std::queue
#  define AUX_STRUCT_TOP  front
#  define AUX_STRUCT_POP  pop
#  define AUX_STRUCT_PUSH push
#elif ORDER_TYPE == DEPTH_FIRST_ORDER
#  include <stack>
#  define AUX_STRUCT std::stack
#  define AUX_STRUCT_TOP  top
#  define AUX_STRUCT_POP  pop
#  define AUX_STRUCT_PUSH push
#endif

#include <numeric>
#include <stdexcept>
#include <iostream>

namespace libheom {

template<typename T>
T calc_gcd(T m, T n) {
  if (m < n) {
    T swap = m;
    m = n;
    n = swap;
  }
  while (n != 0) {
    T swap = n;
    n = m%n;
    m = swap;
  }
  return m;
}


template<typename T>
T calc_multicombination(T n, T r) {
  T num, den;
  num = 1;
  den = 1;
  for(T i = 1; i <= r; ++i) {
    num *= n + i - 1;
    den *= i;
    T gcd = calc_gcd(num, den);
    num /= gcd;
    den /= gcd;
  }
  return num/den;
}

long long calc_hierarchy_element_count(int level,
                                    int dim) {
  return calc_multicombination<long long>(dim + 1, level);
}

#if ORDER_TYPE == ORIGINAL_ORDER
void set_hierarchy_space_sub(hierarchy_space& hs,
                             std::vector<int>& index,
                             int& lidx,
                             int k,
                             int depth,
                             int max_depth,
                             std::function<void(int)> callback,
                             int interval_callback,
                             std::function<bool(std::vector<int>, int)> filter_predicator) {
  for (int j_k = 0; j_k <= max_depth - depth; ++j_k) {
    index[k] = j_k;
    int depth_current = j_k + depth;
    bool pass = (depth <= max_depth) && filter_predicator(index, depth);
    if (pass) {
      if (k == 0) {
        hs.index_book[index] = lidx;
        hs.j.push_back(index);
        ++lidx;
      } else {
        set_hierarchy_space_sub(hs,
                                index,
                                lidx,
                                k - 1,
                                depth_current,
                                max_depth,
                                callback,
                                interval_callback,
                                filter_predicator);
      }
    }
  }
  index[k] = 0;
}
#endif

int allocate_hierarchy_space(hierarchy_space& hs,
                             int max_depth,
                             std::function<void(double)> callback,
                             int interval_callback,
                             std::function<bool(std::vector<int>, int)> filter_predicator) {
  int n_dim = hs.n_dim;
  
  std::vector<int> index(n_dim);
  int n_hierarchy   = 0;
  int lidx          = 0;
#if (ORDER_TYPE == BREADTH_FIRST_ORDER) || (ORDER_TYPE == DEPTH_FIRST_ORDER)
  AUX_STRUCT<std::vector<int>> next_element;
  AUX_STRUCT<int>              k_last_modified;
  int last_modified = 0;
#endif  

  std::fill(index.begin(), index.end(), 0);
#if   ORDER_TYPE == ORIGINAL_ORDER
  set_hierarchy_space_sub(hs, index, lidx, n_dim - 1, 0, max_depth, callback, interval_callback, filter_predicator);
#elif (ORDER_TYPE == BREADTH_FIRST_ORDER) || (ORDER_TYPE == DEPTH_FIRST_ORDER)
  next_element.AUX_STRUCT_PUSH(index);
  k_last_modified.AUX_STRUCT_PUSH(last_modified);
  long long estimated_max_lidx = calc_hierarchy_element_count(max_depth, n_dim);
  
  while (!next_element.empty()) {
    if (lidx % interval_callback == 0) {
      callback(lidx/static_cast<double>(estimated_max_lidx));
    }
    index = next_element.AUX_STRUCT_TOP();
    next_element.AUX_STRUCT_POP();
    last_modified = k_last_modified.AUX_STRUCT_TOP();
    k_last_modified.AUX_STRUCT_POP();
    
    hs.index_book[index] = lidx;
    hs.j.push_back(index);
    ++lidx;
#  if   ORDER_TYPE == BREADTH_FIRST_ORDER
    for (int k = last_modified; k < n_dim; ++k) {
#  elif ORDER_TYPE == DEPTH_FIRST_ORDER
    for (int k = n_dim - 1; k >= last_modified; --k) {
#  endif
        ++index[k];
        int depth = std::accumulate(index.begin(), index.end(), 0);
        bool pass = (depth <= max_depth) && filter_predicator(index, depth);
        if (pass) {
          next_element.AUX_STRUCT_PUSH(index);
          k_last_modified.AUX_STRUCT_PUSH(k);
        }
        --index[k];
#  if   ORDER_TYPE == BREADTH_FIRST_ORDER
    }
#  elif ORDER_TYPE == DEPTH_FIRST_ORDER
    }
#  endif
  }
#endif
  n_hierarchy = lidx;
  hs.ptr_void = lidx;
  
  hs.ptr_p1.resize(n_hierarchy);
  hs.ptr_m1.resize(n_hierarchy);
  for (int lidx = 0; lidx < n_hierarchy; ++lidx) {
    index = hs.j[lidx];
    hs.ptr_p1[lidx].resize(n_dim);
    hs.ptr_m1[lidx].resize(n_dim);
    
    for (int k = 0; k < n_dim; ++k) {
      ++index[k];
      try {
        hs.ptr_p1[lidx][k] = hs.index_book.at(index);
      } catch (std::out_of_range&) {
        hs.ptr_p1[lidx][k] = hs.ptr_void;
      }
      index[k] -= 2;
      try {
        hs.ptr_m1[lidx][k] = hs.index_book.at(index);
      } catch (std::out_of_range&) {
        hs.ptr_m1[lidx][k] = hs.ptr_void;
      }
      ++index[k];
    }
  }
  return n_hierarchy;
}

}
