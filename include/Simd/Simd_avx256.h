/**
 * @file Simd_avx256.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-12
 * 
 * @copyright Copyright (c) 2023
 * 
 */

// clang-format off

#pragma once

#pragma message(" \"Simd_avx256.h\" included");

// #include <avxintrin.h>
// #include <avx2intrin.h>
// #include <fmaintrin.h>
#include <immintrin.h>


// pack
template <typename Tp> struct vReal;
template <> struct vReal<float>  { __m256  vec;  enum { NumElem = 8 }; };
template <> struct vReal<double> { __m256d vec;  enum { NumElem = 4 }; };

// typedef \a vRealF and \a vRealD
typedef vReal<float>  vRealF;
typedef vReal<double> vRealD;

/**
 * @brief \todo Define arithmetic operators: overload + - * () operater
 * 
 */

// Define arithmetic operation
// load
static inline vRealF SimdLoad(const float  *_p) { return {_mm256_load_ps(_p)};}
static inline vRealD SimdLoad(const double *_p) { return {_mm256_load_pd(_p)};}

static inline void SimdLoad(vReal<float>  &a, const float  *_p) { a.vec = _mm256_load_ps(_p);}
static inline void SimdLoad(vReal<double> &a, const double *_p) { a.vec = _mm256_load_pd(_p);}

// store
static inline void SimdStore(float  *_p, const vRealF &a) { _mm256_store_ps(_p, a.vec); }
static inline void SimdStore(double *_p, const vRealD &a) { _mm256_store_pd(_p, a.vec); }

// setzero
static inline void SimdSetzero( vRealF &a) { a.vec = _mm256_setzero_ps(); }
static inline void SimdSetzero( vRealD &a) { a.vec = _mm256_setzero_pd(); }
// static inline vRealF SimdSetzero() { return {_mm256_setzero_ps()}; } // wrong overloading
// static inline vRealD SimdSetzero() { return {_mm256_setzero_pd()}; } // wrong overloading

// add
static inline vRealF SimdAdd(const vRealF &a, const vRealF &b) { return {_mm256_add_ps(a.vec, b.vec)}; }
static inline vRealD SimdAdd(const vRealD &a, const vRealD &b) { return {_mm256_add_pd(a.vec, b.vec)}; }

// sub
static inline vRealF SimdSub(vRealF a, vRealF b) { return {_mm256_sub_ps(a.vec, b.vec)}; }
static inline vRealD SimdSub(vRealD a, vRealD b) { return {_mm256_sub_pd(a.vec, b.vec)}; }

// mul
static inline vRealF SimdMul(vRealF a, vRealF b) { return {_mm256_mul_ps(a.vec, b.vec)}; }
static inline vRealD SimdMul(vRealD a, vRealD b) { return {_mm256_mul_pd(a.vec, b.vec)}; }

// fmadd  dst = a*b+c
static inline vRealF SimdFmadd(vRealF a, vRealF b, vRealF c) { return {_mm256_fmadd_ps(a.vec, b.vec, c.vec)}; } 
static inline vRealD SimdFmadd(vRealD a, vRealD b, vRealD c) { return {_mm256_fmadd_pd(a.vec, b.vec, c.vec)}; }

// fmsub  dst = a*b-c
static inline vRealF SimdFmsub(vRealF a, vRealF b, vRealF c) { return {_mm256_fmsub_ps(a.vec, b.vec, c.vec)}; }
static inline vRealD SimdFmsub(vRealD a, vRealD b, vRealD c) { return {_mm256_fmsub_pd(a.vec, b.vec, c.vec)}; }
