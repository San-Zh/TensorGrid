/**
 * @file SimdTypes.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-10
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include "Simd_vComplex.h"

// clang-format off
template <typename Tp> class vRealTraits;
template <> class vRealTraits<float>  { typedef vRealF value_type; };
template <> class vRealTraits<double> { typedef vRealD value_type; };

template <class vtype> class SimdTraits;
template <> class SimdTraits<vRealF>    { typedef vRealF    value_type; };
template <> class SimdTraits<vRealD>    { typedef vRealD    value_type; };
template <> class SimdTraits<vComplexF> { typedef vComplexF value_type; };
template <> class SimdTraits<vComplexD> { typedef vComplexD value_type; };
// clang-format on

template <class vtype, int N>
class iVector {
  public:
    typedef typename SimdTraits<vtype>::value_type value_type;
    value_type Vec[N];
};

template <class vtype, int M, int N = M>
class iMatrix {
    typedef typename SimdTraits<vtype>::value_type value_type;
    value_type Mat[M][N];
};

template <typename vtype, int M, int N>
auto VVM_Fmadd(iVector<vtype, M> &va, iVector<vtype, N> &vb, iMatrix<vtype, M, N> &m)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            m.Mat[i][j] = SimdFmadd(a.Vec[i], b.Vec[j], m.Mat[i][j]);
        }
    }
}