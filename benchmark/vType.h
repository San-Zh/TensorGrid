/**
 * @file vType.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-08
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include <immintrin.h>

#ifndef ENABLE_SIMD
typedef float  vRealF;
typedef double vRealD;
#endif

#if defined(SSE2)
typedef __m128  vRealF;
typedef __m128d vRealD;
#endif

#if defined(AVX1) || defined(AVX2)
typedef __m256  vRealF;
typedef __m256d vRealD;
#endif

// create C++ vector data type classes,
//     vRealF, vRealD, vComplexF, vComplexD and vInteger.

// make use of C++11 features to reduce the volume of code and increase the flexibility  compared to QDP++.
// We implement internal template classes representing Scalar,Vector or Matrix of anything.
