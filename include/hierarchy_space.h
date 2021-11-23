/*
 * LibHEOM: Copyright (c) Tatsushi Ikeda
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef HIERARCHY_SPACE_H
#define HIERARCHY_SPACE_H

#include <vector>
#include <map>
#include <functional>

namespace libheom {

class hierarchy_space {
public:
  int n_dim;
  std::vector<std::vector<int>> j;
  std::vector<std::vector<int>> ptr_p1;
  std::vector<std::vector<int>> ptr_m1;
  std::map<std::vector<int>,int> index_book;
  int ptr_void;
};

int allocate_hierarchy_space(hierarchy_space& hs,
                             int max_depth,
                             std::function<void(int, int)> callback
                             = [](int, int) { return; },
                             int interval_callback = 1024,
                             std::function<bool(std::vector<int>, int)> hierarchy_filter
                             = [](std::vector<int> index, int depth) -> bool { return true; },
                             bool filter_flag = false);

}

#endif /* HIERARCHY_SPACE_H */
