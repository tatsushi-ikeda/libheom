/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#include "hrchy_space.h"

#define ORIGINAL_ORDER      0
#define BREADTH_FIRST_ORDER 1
#define DEPTH_FIRST_ORDER   2
#define ORDER_TYPE BREADTH_FIRST_ORDER

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


#if ORDER_TYPE == ORIGINAL_ORDER
void set_hrchy_space_sub(hrchy_space& hs,
                         vector<int>& index,
                         int& lidx,
                         int  k,
                         int  depth,
                         int  max_depth,
                         std::function<void(int, int)> callback,
                         int  interval_callback,
                         int  estimated_max_lidx,
                         std::function<bool(vector<int>, int)> hrchy_filter,
                         bool filter_flag)
{
  for (int n_k = 0; n_k <= max_depth - depth; ++n_k) {
    index[k] = n_k;
    int depth_current = n_k + depth;
    bool pass = (depth <= max_depth) && (!filter_flag || hrchy_filter(index, depth));
    if (pass) {
      if (k == 0) {
        if (lidx % interval_callback == 0) {
          callback(lidx, estimated_max_lidx);
        }
        if (depth == max_depth && filter_flag) {
          std::cerr << "[Warning]: hrchy_filter has reached max_depth ";
          print_index(index, std::cerr);
          std::cerr << std::endl; 
        }
        hs.book[index] = lidx;
        hs.n.push_back(index);
        ++lidx;
      } else {
        set_hrchy_space_sub(hs,
                            index,
                            lidx,
                            k - 1,
                            depth_current,
                            max_depth,
                            callback,
                            interval_callback,
                            estimated_max_lidx,
                            hrchy_filter,
                            filter_flag);
      }
    }
  }
  index[k] = 0;
}
#endif


int alloc_hrchy_space(hrchy_space& hs,
                      int  max_depth,
                      std::function<void(int, int)> callback,
                      int  interval_callback,
                      std::function<bool(vector<int>, int)> hrchy_filter,
                      bool filter_flag)
{
  int n_dim = hs.n_dim;
  
  vector<int> index(n_dim);
  int n_hrchy   = 0;
  int lidx          = 0;
#if (ORDER_TYPE == BREADTH_FIRST_ORDER) || (ORDER_TYPE == DEPTH_FIRST_ORDER)
  AUX_STRUCT<vector<int>> next_element;
  AUX_STRUCT<int>         k_last_modified;
  int last_modified = 0;
#endif  

  long long estimated_max_lidx = calc_hrchy_element_count(max_depth, n_dim);
  std::fill(index.begin(), index.end(), 0);
#if   ORDER_TYPE == ORIGINAL_ORDER
  set_hrchy_space_sub(hs, index, lidx, n_dim - 1, 0, max_depth, callback, interval_callback, estimated_max_lidx, hrchy_filter, filter_flag);
#elif (ORDER_TYPE == BREADTH_FIRST_ORDER) || (ORDER_TYPE == DEPTH_FIRST_ORDER)
  next_element.AUX_STRUCT_PUSH(index);
  k_last_modified.AUX_STRUCT_PUSH(last_modified);
  
  while (!next_element.empty()) {
    if (lidx % interval_callback == 0) {
      callback(lidx, estimated_max_lidx);
    }
    index = next_element.AUX_STRUCT_TOP();
    next_element.AUX_STRUCT_POP();
    last_modified = k_last_modified.AUX_STRUCT_TOP();
    k_last_modified.AUX_STRUCT_POP();
    
    hs.book[index] = lidx;
    hs.n.push_back(index);
    ++lidx;
#  if   ORDER_TYPE == BREADTH_FIRST_ORDER
    for (int k = last_modified; k < n_dim; ++k) {
#  elif ORDER_TYPE == DEPTH_FIRST_ORDER
    for (int k = n_dim - 1; k >= last_modified; --k) {
#  endif
        ++index[k];
        int depth = std::accumulate(index.begin(), index.end(), 0);
        bool pass = (depth <= max_depth) && (!filter_flag || hrchy_filter(index, depth));
        if (pass) {
          if (depth == max_depth && filter_flag) {
            std::cerr << "[Warning]: hrchy_filter has reached max_depth ";
            print_index(index, std::cerr);
            std::cerr << std::endl; 
         }
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
  n_hrchy = lidx;
  hs.ptr_void = lidx;
  
  hs.ptr_p1.resize(n_hrchy + 1);
  hs.ptr_m1.resize(n_hrchy + 1);
  for (int lidx = 0; lidx < n_hrchy; ++lidx) {
    index = hs.n[lidx];
    hs.ptr_p1[lidx].resize(n_dim);
    hs.ptr_m1[lidx].resize(n_dim);
    
    for (int k = 0; k < n_dim; ++k) {
      ++index[k];
      try {
        hs.ptr_p1[lidx][k] = hs.book.at(index);
      } catch (std::out_of_range&) {
        hs.ptr_p1[lidx][k] = hs.ptr_void;
      }
      index[k] -= 2;
      try {
        hs.ptr_m1[lidx][k] = hs.book.at(index);
      } catch (std::out_of_range&) {
        hs.ptr_m1[lidx][k] = hs.ptr_void;
      }
      ++index[k];
    }
  }
  hs.ptr_p1[hs.ptr_void].resize(n_dim);
  std::fill(hs.ptr_p1[hs.ptr_void].begin(), hs.ptr_p1[hs.ptr_void].end(), hs.ptr_void);
  hs.ptr_m1[hs.ptr_void].resize(n_dim);
  std::fill(hs.ptr_m1[hs.ptr_void].begin(), hs.ptr_m1[hs.ptr_void].end(), hs.ptr_void);
  return n_hrchy;
}

}
