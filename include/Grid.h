/**
 * @file Grid.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-05-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once
#include "Geometry.h"
#include "Memory.h"
#include "utils.h"

template <typename Tp, size_t GD, MemLevel_t MmLv = MemLevel0>
class Grid {
 public:
  typedef Tp DataType;
  typedef Tp* DataPoiner;
  typedef Geometry<GD> GridParams;
  enum { GridAxes = GD };

 public:
  Grid(const GridParams& g) : _grid(g) {
    std::cout << g.volume() << std::endl;
    if (!_memflag) MemoryAlloc(_data, g.volume() * sizeof(Tp), MmLv);
    _memflag = true;
  };

  ~Grid() {
    if (_memflag) MemoryFree(_data, MmLv);
  };

 private:
  GridParams _grid;
  DataPoiner _data;
  bool _memflag = false;
};
