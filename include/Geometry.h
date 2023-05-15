/**
 * @file Geometry.h
 * @author your name (you@domain.com)
 * @brief stack vector
 * @version 0.1
 * @date 2023-03-09
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once

#include <iostream>
#include <cstring>
#include "utils.h"

using std::size_t;

/**
 * @brief A Geometry, with a n-vector, to declare a geometric structure, and the n-th elem gives the length of the n-th axis.
 * 
 * @tparam NumAxes the number of axes of the Geometry
 */
template <size_t NumAxes>
class Geometry {
  public:
    typedef size_t DataType;
    typedef DataType *pointer;
    typedef DataType DataVect[NumAxes];
    enum { numAxes = NumAxes };
    // typedef size_t VolumeType;

  public:
    ///// This constructor is allowed to create a Greometry<NumAexs> with a vector Vec[N], where N < NumAxes,
    ///// and it will bring some confused length (value) of the n-th (n>=N) axis.
    // _INLINE_ Geometry(const DataType *_vec)
    // {
    //     for (size_t i = 0; i < NumAxes; i++) {
    //         _data[i] = _vec[i];
    //     }
    //     std::cout << "Geometry(const DataType *_vec)" << std::endl;
    // }

    _INLINE_ Geometry(const DataVect &_vec)
    {
        for (size_t i = 0; i < NumAxes; i++) {
            _data[i] = _vec[i];
        }
        // memcpy(this->_data, _vec, sizeof(DataType) * NumAxes);
        // std::cout << "Geometry(const DataVect &_vec)" << std::endl;
    }

    _INLINE_ Geometry(const Geometry<NumAxes> &_vec)
    {
        for (size_t i = 0; i < NumAxes; i++) {
            this->_data[i] = _vec[i];
        }
        // memcpy(this, &_vec, sizeof(*this));
        // std::cout << "Geometry(const Geometry<NumAxes> &_vec)" << std::endl;
    }

    _INLINE_ Geometry(const DataType &_a)
    {
        std::fill(_data, _data + NumAxes, _a);
        // std::cout << "Geometry(const DataType &_a)" << std::endl;
    }

    // _INLINE_ Geometry() = default;
    _INLINE_ ~Geometry() = default;

  public:
    _INLINE_ DataType &operator[](size_t i) { return _data[i]; }
    _INLINE_ const DataType &operator[](size_t i) const { return _data[i]; }

    _INLINE_ void resize(const Geometry &_vec) noexcept
    {
        for (size_t i = 0; i < NumAxes; i++) {
            _data[i] = _vec[i];
        }
    }

    _INLINE_ void resize(const DataVect &_vec) noexcept
    {
        for (size_t i = 0; i < NumAxes; i++) {
            _data[i] = _vec[i];
        }
    }

    _INLINE_ DataType volume() noexcept {
      size_t _volume = 1;
      for (size_t i = 0; i < NumAxes; i++) {
        _volume *= _data[i];
      }
      return _volume;
    }

    _INLINE_ const DataType volume() const noexcept {
      size_t _volume = 1;
      for (size_t i = 0; i < NumAxes; i++) {
        _volume *= _data[i];
      }
      return _volume;
    }

  private:
    DataType _data[NumAxes] = {0};
};

template <size_t NumAxes>
std::ostream &operator<<(std::ostream &os, const Geometry<NumAxes> &vec)
{
    os << "{";
    for (size_t i = 0; i < NumAxes - 1; i++) {
        os << vec[i] << ", ";
    }
    os << vec[NumAxes - 1] << "}";
    return os;
}
