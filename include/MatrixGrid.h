/**
 * @file MatrixGrid.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-11
 * 
 * @copyright Ncolopyright (c) 2023
 * 
 */

#pragma once

#include "utils.h"
#include "Memory.h"
#include "Geometry.h"

/**
 * @brief MatrixGrid
 * 
 * @tparam Tp data type
 * @tparam GridType 
 * @tparam Nrow num of row
 * @tparam Ncol num column
 */
template <typename Tp, size_t Nrow, size_t Ncol, typename GridType>
class MatrixGrid {
  public:
    typedef Tp Datatype;
    typedef Tp *DataPointer;
    // typedef Geometry<D> GridType;
    enum { griddim = GridType::numAxes };

  public:
    MatrixGrid(const GridType &g) : _grid(g)
    {
        _volume = _grid.volume();
        _size = _grid.volume() * Ncol * Nrow;
        MemoryAlloc(_data, _size * sizeof(Tp));
        _memflag = true;
        repermute();
    };

    ~MatrixGrid()
    {
        if (_memflag)
            MemoryFree(_data);
    };

    _INLINE_ const bool state() const { return _memflag; }
    _INLINE_ const size_t &size() const { return _size; }
    _INLINE_ const size_t &volume() const { return _volume; }
    _INLINE_ const GridType &girdstructure() { return _grid; }
    // _INLINE_ constexpr size_t griddim() const { return _grid.axes(); }

    _INLINE_ DataPointer data() { return _data; }

    /// @brief  to sequentially access to memory by grid-sized axes;
    /// @param n the n-th grid-sized memory;
    /// @return (Tp *) pointer to the n-th grid-sized memory;
    _INLINE_ DataPointer datablcok(size_t n) { return _data + n * _volume; }

    /// @brief to access to the memory of [ns][nc]-th grid with a deifned pointer
    /// @param ns
    /// @param nc
    /// @return (Tp *) pointer
    _INLINE_ DataPointer &grid(size_t ns, size_t nc) { return _grid[ns][nc][0]; }
    _INLINE_ DataPointer &gridreal(size_t ns, size_t nc) { return grid[ns][nc][0]; }
    _INLINE_ DataPointer &gridimag(size_t ns, size_t nc) { return grid[ns][nc][1]; }

    _INLINE_ void resize(const GridType &g)
    {
        if (_memflag)
            MemoryFree(_data);
        _grid = g;
        _volume = _grid.volume();
        _size = _grid.volume() * Ncol * Nrow;
        MemoryAlloc(_data, _size * sizeof(Tp));
        _memflag = true;
        repermute();
    }

    _INLINE_ void repermute()
    {
        if (_memflag) {
            for (size_t _s = 0; _s < Nrow; _s++) {
                for (size_t _c = 0; _c < Ncol; _c++) {
                    _gptr[_s][_c][0] = _data + 2 * (_s * Ncol + _c);
                    _gptr[_s][_c][1] = _data + 2 * (_s * Ncol + _c) + 1;
                }
            }
        } else {
            std::cout << "error: _memflag = 0, i.e, MatrixGrid is not alloced!" << std::endl;
        }
    }

  private:
    const static size_t _griddim = GridType::numAxes;
    GridType _grid;
    size_t _volume;
    size_t _size;
    DataPointer _data;
    DataPointer _gptr[Nrow][Ncol][2];
    bool _memflag = false;
    // DataPointer _gridptr[Nrow][Ncol][2];
};
