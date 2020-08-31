/*
 * LibHEOM, version 0.5
 * Copyright (c) 2019-2020 Tatsushi Ikeda
 *
 * This library is distributed under BSD 3-Clause License.
 * See LINCENSE.txt for licence.
 *------------------------------------------------------------------------*/

#ifndef HIERARCHY_SPACE_H
#define HIERARCHY_SPACE_H

#include <vector>
#include <map>
#include <functional>

namespace libheom {

class HierarchySpace {
public:
  int n_dim;
  std::vector<std::vector<int>> j;
  std::vector<std::vector<int>> ptr_p1;
  std::vector<std::vector<int>> ptr_m1;
  std::map<std::vector<int>,int> index_book;
  int ptr_void;
};

int AllocateHierarchySpace(HierarchySpace& hs,
                           int max_depth,
                           std::function<void(double)> callback
                           = [](int) { return; },
                           int interval_callback = 1024,
                           std::function<bool(std::vector<int>, int)> filter_predicator
                           = [](std::vector<int> index, int depth) -> bool { return true; });

}

#endif /* HIERARCHY_SPACE_H */
