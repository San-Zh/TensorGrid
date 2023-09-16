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

#pragma message(" \"Simd_avx256.h\" included")

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

// typedef
typedef vReal<float>  vRealF;
typedef vReal<double> vRealD;

///////////////// with a void type return; ////////////////////
// load
static inline void SimdLoad(vReal<float>  &a, const float  *_p) { a.vec = _mm256_load_ps(_p);}
static inline void SimdLoad(vReal<double> &a, const double *_p) { a.vec = _mm256_load_pd(_p);}

// store
static inline void SimdStore(float  *_p, const vRealF &a) { _mm256_store_ps(_p, a.vec); }
static inline void SimdStore(double *_p, const vRealD &a) { _mm256_store_pd(_p, a.vec); }

// setzero
static inline void SimdSetzero( vRealF &a) { a.vec = _mm256_setzero_ps(); }
static inline void SimdSetzero( vRealD &a) { a.vec = _mm256_setzero_pd(); }

// add
static inline void SimdAdd(vRealF &ret, const vRealF &a, const vRealF &b) { ret.vec =_mm256_add_ps(a.vec, b.vec); }
static inline void SimdAdd(vRealD &ret, const vRealD &a, const vRealD &b) { ret.vec =_mm256_add_pd(a.vec, b.vec); }

// sub
static inline void SimdSub(vRealF &ret,const  vRealF a, const vRealF b) { ret.vec = _mm256_sub_ps(a.vec, b.vec); }
static inline void SimdSub(vRealD &ret,const  vRealD a, const vRealD b) { ret.vec = _mm256_sub_pd(a.vec, b.vec); }

// mul
static inline void SimdMul(vRealF &ret,const  vRealF a, const vRealF b) { ret.vec = _mm256_mul_ps(a.vec, b.vec); }
static inline void SimdMul(vRealD &ret,const  vRealD a, const vRealD b) { ret.vec = _mm256_mul_pd(a.vec, b.vec); }

// fmadd  dst = a*b+c
static inline void SimdFmadd(vRealF &ret, const vRealF &a, const vRealF &b, const vRealF &c) { ret.vec = _mm256_fmadd_ps(a.vec, b.vec, c.vec);}  
static inline void SimdFmadd(vRealD &ret, const vRealD &a, const vRealD &b, const vRealD &c) { ret.vec = _mm256_fmadd_pd(a.vec, b.vec, c.vec);} 

// fmsub  dst = a*b-c
static inline void SimdFmsub(vRealF &ret, const vRealF &a, const vRealF &b, const vRealF &c) { ret.vec = _mm256_fmsub_ps(a.vec, b.vec, c.vec); }
static inline void SimdFmsub(vRealD &ret, const vRealD &a, const vRealD &b, const vRealD &c) { ret.vec = _mm256_fmsub_pd(a.vec, b.vec, c.vec); }


//////////////////// with a vReal<Tp> type return; ///////////////////

// load
static inline vRealF SimdLoad(const float  *_p) { return {_mm256_load_ps(_p)};}
static inline vRealD SimdLoad(const double *_p) { return {_mm256_load_pd(_p)};}

// static inline vRealF SimdSetzero() { return {_mm256_setzero_ps()}; } // wrong overloading
// static inline vRealD SimdSetzero() { return {_mm256_setzero_pd()}; } // wrong overloading

// add
static inline vRealF SimdAdd(const vRealF &a, const vRealF &b) { return {_mm256_add_ps(a.vec, b.vec)}; }
static inline vRealD SimdAdd(const vRealD &a, const vRealD &b) { return {_mm256_add_pd(a.vec, b.vec)}; }

// sub
static inline vRealF SimdSub(const vRealF &a, const vRealF &b) { return {_mm256_sub_ps(a.vec, b.vec)}; }
static inline vRealD SimdSub(const vRealD &a, const vRealD &b) { return {_mm256_sub_pd(a.vec, b.vec)}; }

// mul
static inline vRealF SimdMul(const vRealF &a, const vRealF &b) { return {_mm256_mul_ps(a.vec, b.vec)}; }
static inline vRealD SimdMul(const vRealD &a, const vRealD &b) { return {_mm256_mul_pd(a.vec, b.vec)}; }

// fmadd  dst = a*b+c
static inline vRealF SimdFmadd(const vRealF &a,const  vRealF &b, const vRealF &c) { return {_mm256_fmadd_ps(a.vec, b.vec, c.vec)}; } 
static inline vRealD SimdFmadd(const vRealD &a,const  vRealD &b, const vRealD &c) { return {_mm256_fmadd_pd(a.vec, b.vec, c.vec)}; }

// fmsub  dst = a*b-c
static inline vRealF SimdFmsub(const vRealF &a, const vRealF &b, const vRealF &c) { return {_mm256_fmsub_ps(a.vec, b.vec, c.vec)}; }
static inline vRealD SimdFmsub(const vRealD &a, const vRealD &b, const vRealD &c) { return {_mm256_fmsub_pd(a.vec, b.vec, c.vec)}; }

// clang-format on