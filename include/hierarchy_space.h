/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LICENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef LIBHEOM_HIERARCHY_SPACE_H
#define LIBHEOM_HIERARCHY_SPACE_H

#include <map>
#include <functional>
#include "type.h"

namespace libheom
{

class hierarchy_space
{
 public:
  int n_modes;
  vector<vector<int>> n;
  vector<vector<int>> ptr_p1;
  vector<vector<int>> ptr_m1;
  std::map<vector<int>,int> multi_index_map;
  int ptr_void;
};

int alloc_hierarchy_space(hierarchy_space& hs,
                      int truncation_depth,
                      std::function<void(int, int)> callback
                      = [](int, int) { return; },
                      int interval_callback = 1024,
                      std::function<bool(vector<int>, int)> hierarchy_filter
                      = [](vector<int> index, int depth) -> bool { return true; },
                      bool filter_flag = false);

}

#endif /* HRCHY_SPACE_H */
