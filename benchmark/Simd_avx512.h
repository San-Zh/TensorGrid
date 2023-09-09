/**
 * @file Simd_avx512.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-09
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#if defined(ENABLE_SIMD) && defined(AVX512F)

#pragma once

#include <immintrin.h>

// #define _inline_ static inline

typedef __m512  vRealF;
typedef __m512d vRealD;

// load
static inline auto SimdLoad(const float *_p) { return _mm512_load_ps(_p); }
static inline auto SimdLoad(const double *_p) { return _mm512_load_pd(_p); }

// store
static inline void SimdStore(void *_p, vRealF a) { _mm512_store_ps(_p, a); }
static inline void SimdStore(void *_p, vRealD a) { _mm512_store_pd(_p, a); }

// add
static inline vRealF SimdAdd(vRealF a, vRealF b) { return _mm512_add_ps(a, b); }
static inline vRealD SimdAdd(vRealD a, vRealD b) { return _mm512_add_pd(a, b); }

// sub
static inline vRealF SimdSub(vRealF a, vRealF b) { return _mm512_sub_ps(a, b); }
static inline vRealD SimdSub(vRealD a, vRealD b) { return _mm512_sub_pd(a, b); }

// mul
static inline vRealF SimdMul(vRealF a, vRealF b) { return _mm512_mul_ps(a, b); }
static inline vRealD SimdMul(vRealD a, vRealD b) { return _mm512_mul_pd(a, b); }

// fmadd  dst = a*b+c
static inline vRealF SimdFmadd(vRealF a, vRealF b, vRealF c) { return _mm512_fmadd_ps(a, b, c); }
static inline vRealD SimdFmadd(vRealD a, vRealD b, vRealD c) { return _mm512_fmadd_pd(a, b, c); }

// fmsub  dst = a*b-c
static inline vRealF SimdFmsub(vRealF a, vRealF b, vRealF c) { return _mm512_fmsub_ps(a, b, c); }
static inline vRealD SimdFmsub(vRealD a, vRealD b, vRealD c) { return _mm512_fmsub_pd(a, b, c); }

#endif