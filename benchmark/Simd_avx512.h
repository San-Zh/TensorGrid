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

#pragma once

#include <immintrin.h>

// typedef __m512  vRealF;
// typedef __m512d vRealD;

// clang-format off
template <typename Tp> struct vReal;
template <> struct vReal<float>;
template <> struct vReal<double>;

template <> 
struct vReal<float>
{
    __m512 vec;
    // void load(const float *_p) { vec = _mm512_load_ps(_p); }
    // void store(float *_p) { _mm512_store_ps(_p, vec); }
    // void setzero() { vec = _mm512_setzero_ps(); }
};

template <> 
struct vReal<double> 
{ 
    __m512d vec; 
    // void load(const double *_p) { vec =  _mm512_load_pd(_p); }
    // void store(double *_p) { _mm512_store_pd(_p, vec); }
    // void setzero() { vec = _mm512_setzero_pd(); }
};

// tyoedef
typedef vReal<float>  vRealF;
typedef vReal<double> vRealD;

// load
// template <typename Tp> void SimdLoad(vReal<Tp> &a,  const Tp * _p);
static inline vRealF SimdLoad(const float  *_p) { return {_mm512_load_ps(_p)};}
static inline vRealD SimdLoad(const double *_p) { return {_mm512_load_pd(_p)};}

static inline void SimdLoad(vReal<float>  &a, const float  *_p) { a.vec = _mm512_load_ps(_p);}
static inline void SimdLoad(vReal<double> &a, const double *_p) { a.vec = _mm512_load_pd(_p);}

// store
static inline void SimdStore(float  *_p, const vRealF &a) { _mm512_store_ps(_p, a.vec); }
static inline void SimdStore(double *_p, const vRealD &a) { _mm512_store_pd(_p, a.vec); }

// setzero
static inline void SimdSetzero( vRealF &a) { a.vec = _mm512_setzero_ps(); }
static inline void SimdSetzero( vRealD &a) { a.vec = _mm512_setzero_pd(); }
// static inline vRealF SimdSetzero() { return {_mm512_setzero_ps()}; } // wrong overloading
// static inline vRealD SimdSetzero() { return {_mm512_setzero_pd()}; } // wrong overloading

// add
static inline vRealF SimdAdd(const vRealF &a, const vRealF &b) { return {_mm512_add_ps(a.vec, b.vec)}; }
static inline vRealD SimdAdd(const vRealD &a, const vRealD &b) { return {_mm512_add_pd(a.vec, b.vec)}; }

// sub
static inline vRealF SimdSub(vRealF a, vRealF b) { return {_mm512_sub_ps(a.vec, b.vec)}; }
static inline vRealD SimdSub(vRealD a, vRealD b) { return {_mm512_sub_pd(a.vec, b.vec)}; }

// mul
static inline vRealF SimdMul(vRealF a, vRealF b) { return {_mm512_mul_ps(a.vec, b.vec)}; }
static inline vRealD SimdMul(vRealD a, vRealD b) { return {_mm512_mul_pd(a.vec, b.vec)}; }

// fmadd  dst = a*b+c
static inline vRealF SimdFmadd(vRealF a, vRealF b, vRealF c) { return {_mm512_fmadd_ps(a.vec, b.vec, c.vec)}; } 
static inline vRealD SimdFmadd(vRealD a, vRealD b, vRealD c) { return {_mm512_fmadd_pd(a.vec, b.vec, c.vec)}; }

// fmsub  dst = a*b-c
static inline vRealF SimdFmsub(vRealF a, vRealF b, vRealF c) { return {_mm512_fmsub_ps(a.vec, b.vec, c.vec)}; }
static inline vRealD SimdFmsub(vRealD a, vRealD b, vRealD c) { return {_mm512_fmsub_pd(a.vec, b.vec, c.vec)}; }

// clang-format on

// template <typename Tp> struct vRealTraits;
// template <> struct vRealTraits<float>  { typedef vReal<Tp> vtype; };
// template <> struct vRealTraits<double> { typedef vReal<Tp> vtype; };
