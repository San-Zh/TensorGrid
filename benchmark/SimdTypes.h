
#pragma once
#include "Simd_vComplex.h"

template <typename Tp>
class vRealTraits;

template <>
class vRealTraits<float> {
  public:
    typedef vRealF value_type;
};

template <>
class vRealTraits<double> {
  public:
    typedef vRealD value_type;
};

template <class vtype>
class SimdTraits;

template <>
class SimdTraits<vRealF> {
    typedef vRealF value_type;
};

template <>
class SimdTraits<vRealD> {
    typedef vComplexD value_type;
};

template <>
class SimdTraits<vComplexF> {
    typedef vComplexF value_type;
};

template <>
class SimdTraits<vComplexD> {
    typedef vComplexD value_type;
};

template <class vtype, int N>
class iVector {
  public:
    typedef typename SimdTraits<vtype>::value_type value_type;
    value_type                                     Vec[N];
};
template <class vtype, int N>
class iMatrix {
    typedef typename SimdTraits<vtype>::value_type value_type;
    value_type                                     Mat[N][N];
};