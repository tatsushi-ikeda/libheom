/* -*- mode:c++ -*-
 * LibHEOM
 * Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef LIBHEOM_HRCHY_SPACE_H
#define LIBHEOM_HRCHY_SPACE_H

#include "type.h"
#include <functional>
#include <map>

namespace libheom {

class hrchy_space {
 public:
  int                        n_dim;
  vector<vector<int>>        n;
  vector<vector<int>>        ptr_p1;
  vector<vector<int>>        ptr_m1;
  std::map<vector<int>, int> book;
  int                        ptr_void;
};

int alloc_hrchy_space(hrchy_space &hs,
                      int max_depth,
                      std::function<void(int, int)> callback
                      = [] (int, int) { return;
                      },
                      int interval_callback = 1024,
                      std::function<bool(vector<int>, int)> hrchy_filter
                                       = [] (vector<int> index, int depth)->bool { return true; },
                      bool filter_flag = false);

} // namespace libheom
#endif /* HRCHY_SPACE_H */
