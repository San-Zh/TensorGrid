/**
 * @file Simd_vtypeTraits.h
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

template <typename SimdVtype> struct Simd_vtypeTraits;

template <>
struct Simd_vtypeTraits<vReal<float>> { typedef vReal<float> vtype;};

template <>
struct Simd_vtypeTraits<vReal<double>> { typedef vReal<double> vtype;};

template <>
struct Simd_vtypeTraits<vComplex<float>> { typedef vComplex<float> vtype;};

template <>
struct Simd_vtypeTraits<vComplex<double>> { typedef vComplex<double> vtype;};

// clang-format on