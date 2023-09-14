/**
 * @file SimdTupleTypes.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-13
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "SimdTraits.h"

template <class vtype, int N>
class iVector {
  public:
    typename SimdTraits<vtype>::vtype vec[N];
};

template <class vtype, int M, int N = M>
class iMatrix {
  public:
    typename SimdTraits<vtype>::vtype mat[M][N];
};

// template <class vtype, int M, int N = M>
// class iMatrix {
//   public:
//     typename SimdTraits<vtype>::value_type mat[M][N];
// };


template <class vtype, int N>
using kerVector = typename SimdTraits<vtype>::vtype[N];


template <class vtype, unsigned M, unsigned N>
using kerMatrix = typename SimdTraits<vtype>::vtype[M][N];