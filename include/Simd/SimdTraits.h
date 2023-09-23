/**
 * @file SimdTraits.h
 * @author your name (you@domain.com)
 * @brief Simd vtype traits, e.g. vReal<float>, vReal<double>, vComplex<float>, vComplex<double>
 * @version 0.1
 * @date 2023-09-12
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include "Simd_opt.h"
#include "Simd_complex.h"
#include "TensorGridEnum.h"

// clang-format off
/**
 * @brief Simd vtype traits for vReal<float>, vReal<double>, vComplex<float>, vComplex<double>
 * \todo \b TODO Another vtype \b vComplex<std::complex<Tp>> shuold be considered.
 * \todo \b TODO \b SimdPrecTraits<PrecisonEnum_t PrecEnum>; PrecEnum:= S,D,C,Z
 * @tparam Simd_vtype 
 */
template <typename Simd_vtype> struct SimdTraits;

// clang-format on
template <>
struct SimdTraits<vReal<float>> {
    typedef vReal<float> vtype;
    typedef float        dtype;
    typedef float       *ptr_type;
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
    typedef double          *ptr_type[2];
    enum { NumElem = vComplex<double>::NumElem };
};


// // clang-format off
// /**
//  * @brief Simd vtype traits, e.g. vReal<float>, vReal<double>, vComplex<float>, vComplex<double>
//  *
//  * @tparam Simd_vtype
//  */
template <Enum_TensorGridPrecision_t PrecEnum>
struct SimdPrecisionTraits;

// // clang-format on
// template <>
// struct SimdTraits<vReal<float>> {
//     typedef vReal<float> vtype;
//     typedef float        dtype;
//     typedef dtype       *ptr_type;
//     enum { NumElem = vReal<float>::NumElem };
// };

// template <>
// struct SimdTraits<vReal<double>> {
//     typedef vReal<double> vtype;
//     typedef double        dtype;
//     typedef double       *ptr_type;
//     enum { NumElem = vReal<double>::NumElem };
// };

// template <>
// struct SimdTraits<vComplex<float>> {
//     typedef vComplex<float> vtype;
//     typedef float           dtype;
//     typedef float          *ptr_type[2];
//     enum { NumElem = vComplex<float>::NumElem };
// };

// template <>
// struct SimdTraits<vComplex<double>> {
//     typedef vComplex<double> vtype;
//     typedef double           dtype;
//     typedef double          *ptr_type[2];
//     enum { NumElem = vComplex<double>::NumElem };
// };



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