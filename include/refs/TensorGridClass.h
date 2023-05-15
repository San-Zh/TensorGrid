/**
 * @file TensorGrid.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-03-09
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "Memory.h"
#include "Geometry.h"
#include "TensorGridEnum.h"

template <int TensorDim>
using TensorStructure = Geometry<TensorDim>;

template <int GridDim>
using GridStructure = Geometry<GridDim>;

template <typename Tp, int GridDim, int TensorDim>
class TensorGrid {
  public:
    // typedef unsigned long uint64_t;
    typedef Tp DataType;
    typedef Tp *DataPointer;

  public:
    TensorGrid(GridStructure<GridDim> G, TensorStructure<TensorDim> T, MemLevel_t ml = MemLevel0)
        : _tensorStructure(T), _gridStructure(G), _gridvolume(G.volume()),
          _tensorvolume(T.volume()), _memlevel(ml), _memflag(true)
    {
        _size = _gridvolume * _tensorvolume;
        if (!_data) {
            MemoryAlloc(_data, _size * sizeof(Tp));
        }
    }
    // TensorGrid(const int (&G)[GridDim], const int (&T)[TensorDim], MemLevel_t memlevel);

    ~TensorGrid()
    {
        if (_memflag) {
            MemoryFree(_data, _memlevel);
            _data = nullptr;
        }
    }

  public:
    auto &tensorStructure() { return _tensorStructure; }
    auto &gridStructure() { return _gridStructure; }
    uint64_t size() const { return _size; }

  private:
    bool _memflag = false;
    MemLevel_t _memlevel = 0;
    DataPointer _data = nullptr;

    uint64_t _size;
    uint64_t _gridvolume;
    uint64_t _tensorvolume;
    GridStructure<GridDim> _gridStructure;
    TensorStructure<TensorDim> _tensorStructure;
};
