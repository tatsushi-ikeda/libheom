/*
 * LibHEOM: Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef HRCHY_SPACE_H
#define HRCHY_SPACE_H

#include <vector>
#include <map>
#include <functional>

namespace libheom
{

class hrchy_space
{
 public:
  int n_dim;
  std::vector<std::vector<int>> n;
  std::vector<std::vector<int>> ptr_p1;
  std::vector<std::vector<int>> ptr_m1;
  std::map<std::vector<int>,int> book;
  int ptr_void;
};


int alloc_hrchy_space
/**/(hrchy_space& hs,
     int max_depth,
     std::function<void(int, int)> callback
     = [](int, int) { return; },
     int interval_callback = 1024,
     std::function<bool(std::vector<int>, int)> hrchy_filter
     = [](std::vector<int> index, int depth) -> bool { return true; },
     bool filter_flag = false);

}

#endif /* HRCHY_SPACE_H */
