/**
 * @file Geometry.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-12
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#if 0

#pragma once

#include "utils.h"
#include "Geometry.h"

/// number of grid dimension
#ifndef GridDim
#define GridDim 4
#endif

template <typename ST>
struct Structure {
};

template <>
struct Structure<Geometry<GridDim>> {
    Geometry<GridDim> coordinate;
    size_t volume = coordinate.volume();
};

using GridStuct = Structure<Geometry<GridDim>>;
using GridStuct4D = Structure<Geometry<4>>;

#endif

// template <int TensorDim>
// using TensorStructure = Geometry<TensorDim>;

// template <int GridDim>
// using GridStructure = Geometry<GridDim>;

// template <int Dimension>
// class GridStructure {
//   public:
//     typedef unsigned int uInt;
//     typedef unsigned int uLong;

//   public:
//     _INLINE_ GridStructure() = default;
//     _INLINE_ GridStructure(const int *Struct) : _griddimen(Dimension)
//     {
//         for (int i = 0; i < TensorRank; i++) {
//             _structure[i] = shape[i];
//         }
//     }
//     ~GridStructure() = default;

//   public:
//     _INLINE_ uInt &dimension() const { return _griddimen; }
//     _INLINE_ uInt &operator[](int i) const { return _structure[i]; }
//     _INLINE_ uLong gridVolume() const
//     {
//         uLong _volume = 1;
//         for (uInt i = 0; i < Dimension; i++) {
//             _volume *= _structure[i]
//         }
//         return _volume;
//     }

//   private:
//     uInt _griddimen;
//     uLong _structure[Dimension];
// };

// template <int TensorRank>
// class TensorStructure {
//   public:
//     typedef unsigned int uInt;
//     typedef unsigned int uLong;

//   public:
//     TensorStructure() = default;
//     _INLINE_ TensorStructure(const uInt *shape) : _tensorrank(TensorRank)
//     {
//         for (uInt i = 0; i < TensorRank; i++) {
//             _structure[i] = shape[i];
//         }
//     }
//     ~TensorStructure() = default;

//   public:
//     _INLINE_ uInt &tensorrank() const { return _tensorrank; }
//     _INLINE_ uInt &operator[](int i) const { return _structure[i]; }
//     _INLINE_ uLong tensorVolume() const
//     {
//         uLong _volume = 1;
//         for (uInt i = 0; i < TensorRank; i++) {
//             _volume *= _structure[i]
//         }
//         return _volume;
//     }

//   private:
//     int _tensorrank;
//     int _structure[TensorRank];
// };
