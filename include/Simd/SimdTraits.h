/**
 * @file SimdTraits.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-12
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include "Simd_opt.h"
#include "Simd_complex.h"

// clang-format off

template <typename Simd_vtype> struct SimdTraits;

// clang-format on
template <>
struct SimdTraits<vReal<float>> {
    typedef vReal<float> vtype;
    typedef float        dtype;
    typedef dtype       *ptr_type;
    enum { NumElem = vReal<float>::NumElem };
};

template <>
struct SimdTraits<vReal<double>> {
    typedef vReal<double> vtype;
    typedef double        dtype;
    typedef double       *ptr_type;
    enum { NumElem = vReal<double>::NumElem };
};

template <>
struct SimdTraits<vComplex<float>> {
    typedef vComplex<float> vtype;
    typedef float           dtype;
    typedef float          *ptr_type[2];
    enum { NumElem = vComplex<float>::NumElem };
};

template <>
struct SimdTraits<vComplex<double>> {
    typedef vComplex<double> vtype;
    typedef double           dtype;
    typedef double         **ptr_type;
    enum { NumElem = vComplex<double>::NumElem };
};


// template <typename Tp>
// using vRealT = typename SimdTraits<Tp>::vtype;

// template <typename Tp>
// using vComplexT = typename SimdTraits<vComplex<Tp>>::vtype;


/// \todo temolate<class Tp> class vComplex<Tp>{} not defined
#if 0
template <>
struct SimdTraits<vComplex<std::complex<float>>> {
    typedef vComplex<std::complex<float>> vtype;
    enum { NumElem = vComplex<std::complex<float>>::NumElem };
};

template <>
struct SimdTraits<vComplex<std::complex<double>>> {
    typedef vComplex<std::complex<double>> vtype;
    enum { NumElem = vComplex<std::complex<double>>::NumElem };
};
#endif