/**
 * @file Simd_avx256.h
 * @author your name (you@domain.com)
 * @brief vReal<float>, vReal<double> for avx, 256 bits simd vector
 * @version 0.1
 * @date 2023-09-12
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#pragma message(" TensorGrid/include/Simd/Simd_sse.h included")

#include <immintrin.h>

// clang-format off

/**
 * @brief vReal<float>, vReal<double> for avx, 256 bits simd vector
 * 
 * @tparam Tp 
 */
template <typename Tp> struct vReal;
template <> struct vReal<float>;
template <> struct vReal<double>;


template <>
struct vReal<float>
{
    __m128 vec;
    enum { NumElem = 4 };

    inline void load(const float *_p, size_t _ofs) { vec = _mm_load_ps(&_p[_ofs]); }
    inline void load(const float *_p) { vec = _mm_load_ps(_p); }

    inline void store(float *_p, size_t _ofs) { _mm_store_ps(&_p[_ofs], vec); }
    inline void store(float *_p) { _mm_store_ps(_p, vec); }

    inline void setzero() { vec = _mm_setzero_ps(); }
    inline void set(const float &_a) { vec = _mm_set1_ps(_a); }
};


template <>
struct vReal<double> 
{
    __m128d vec;
    enum { NumElem = 2 };

    inline void load(const double *_p, size_t _ofs) { vec = _mm_load_pd(&_p[_ofs]); }
    inline void load(const double *_p) { vec = _mm_load_pd(_p); }

    inline void store(double *_p, size_t _ofs) { _mm_store_pd(&_p[_ofs], vec); }
    inline void store(double *_p) { _mm_store_pd(_p, vec); }

    inline void setzero() { vec = _mm_setzero_pd(); }
    inline void set(const double &_a) { vec = _mm_set1_pd(_a); }
};


// typedef
typedef vReal<float>  vRealF;
typedef vReal<double> vRealD;


///////////////// with a void type return; ////////////////////

// load
static inline void SimdLoad(vReal<float>  &a, const float  *_p) { a.vec = _mm_load_ps(_p);}
static inline void SimdLoad(vReal<double> &a, const double *_p) { a.vec = _mm_load_pd(_p);}

// store
static inline void SimdStore(float  *_p, const vRealF &a) { _mm_store_ps(_p, a.vec); }
static inline void SimdStore(double *_p, const vRealD &a) { _mm_store_pd(_p, a.vec); }

static inline void SimdStore(float  *_p,const  size_t v, const vRealF &a) { _mm_store_ps(&_p[v], a.vec); }
static inline void SimdStore(double *_p,const  size_t v, const vRealD &a) { _mm_store_pd(&_p[v], a.vec); }

// setzero
static inline void SimdSetzero( vRealF &a) { a.vec = _mm_setzero_ps(); }
static inline void SimdSetzero( vRealD &a) { a.vec = _mm_setzero_pd(); }

// add
static inline void SimdAdd(vRealF &ret, const vRealF &a, const vRealF &b) { ret.vec =_mm_add_ps(a.vec, b.vec); }
static inline void SimdAdd(vRealD &ret, const vRealD &a, const vRealD &b) { ret.vec =_mm_add_pd(a.vec, b.vec); }

// sub
static inline void SimdSub(vRealF &ret,const  vRealF a, const vRealF b) { ret.vec = _mm_sub_ps(a.vec, b.vec); }
static inline void SimdSub(vRealD &ret,const  vRealD a, const vRealD b) { ret.vec = _mm_sub_pd(a.vec, b.vec); }

// mul
static inline void SimdMul(vRealF &ret,const  vRealF a, const vRealF b) { ret.vec = _mm_mul_ps(a.vec, b.vec); }
static inline void SimdMul(vRealD &ret,const  vRealD a, const vRealD b) { ret.vec = _mm_mul_pd(a.vec, b.vec); }

#if defined(__FMA__)
// fmadd  dst = a*b+c
static inline void SimdFmadd(vRealF &ret, const vRealF &a, const vRealF &b, const vRealF &c) { ret.vec = _mm_fmadd_ps(a.vec, b.vec, c.vec);}  
static inline void SimdFmadd(vRealD &ret, const vRealD &a, const vRealD &b, const vRealD &c) { ret.vec = _mm_fmadd_pd(a.vec, b.vec, c.vec);} 

// fmsub  dst = a*b-c
static inline void SimdFmsub(vRealF &ret, const vRealF &a, const vRealF &b, const vRealF &c) { ret.vec = _mm_fmsub_ps(a.vec, b.vec, c.vec); }
static inline void SimdFmsub(vRealD &ret, const vRealD &a, const vRealD &b, const vRealD &c) { ret.vec = _mm_fmsub_pd(a.vec, b.vec, c.vec); }

#else


// fmadd  dst = a*b+c
static inline void SimdFmadd(vRealF &ret, const vRealF &a,const  vRealF &b, const vRealF &c) { ret.vec = _mm_add_ps( _mm_mul_ps(a.vec, b.vec), c.vec); }
static inline void SimdFmadd(vRealD &ret, const vRealD &a,const  vRealD &b, const vRealD &c) { ret.vec = _mm_add_pd( _mm_mul_pd(a.vec, b.vec), c.vec); }

// fmsub  dst = a*b-c
static inline void SimdFmsub(vRealF &ret, const vRealF &a, const vRealF &b, const vRealF &c) { ret.vec = _mm_sub_ps(_mm_mul_ps(a.vec, b.vec), c.vec); }
static inline void SimdFmsub(vRealD &ret, const vRealD &a, const vRealD &b, const vRealD &c) { ret.vec = _mm_sub_pd(_mm_mul_pd(a.vec, b.vec), c.vec); }


#endif

//////////////////// with a vReal<Tp> type return; ///////////////////

// load
static inline vRealF SimdLoad(const float  *_p) { return {_mm_load_ps(_p)};}
static inline vRealD SimdLoad(const double *_p) { return {_mm_load_pd(_p)};}

// add
static inline vRealF SimdAdd(const vRealF &a, const vRealF &b) { return {_mm_add_ps(a.vec, b.vec)}; }
static inline vRealD SimdAdd(const vRealD &a, const vRealD &b) { return {_mm_add_pd(a.vec, b.vec)}; }

// sub
static inline vRealF SimdSub(const vRealF &a, const vRealF &b) { return {_mm_sub_ps(a.vec, b.vec)}; }
static inline vRealD SimdSub(const vRealD &a, const vRealD &b) { return {_mm_sub_pd(a.vec, b.vec)}; }

// mul
static inline vRealF SimdMul(const vRealF &a, const vRealF &b) { return {_mm_mul_ps(a.vec, b.vec)}; }
static inline vRealD SimdMul(const vRealD &a, const vRealD &b) { return {_mm_mul_pd(a.vec, b.vec)}; }

#if defined(__FMA__)
// fmadd  dst = a*b+c
static inline vRealF SimdFmadd(const vRealF &a,const  vRealF &b, const vRealF &c) { return {_mm_fmadd_ps(a.vec, b.vec, c.vec)}; } 
static inline vRealD SimdFmadd(const vRealD &a,const  vRealD &b, const vRealD &c) { return {_mm_fmadd_pd(a.vec, b.vec, c.vec)}; }

// fmsub  dst = a*b-c
static inline vRealF SimdFmsub(const vRealF &a, const vRealF &b, const vRealF &c) { return {_mm_fmsub_ps(a.vec, b.vec, c.vec)}; }
static inline vRealD SimdFmsub(const vRealD &a, const vRealD &b, const vRealD &c) { return {_mm_fmsub_pd(a.vec, b.vec, c.vec)}; }

#else

// fmadd  dst = a*b+c
static inline vRealF SimdFmadd(const vRealF &a,const  vRealF &b, const vRealF &c) { return {_mm_add_ps( _mm_mul_ps(a.vec, b.vec), c.vec)}; } 
static inline vRealD SimdFmadd(const vRealD &a,const  vRealD &b, const vRealD &c) { return {_mm_add_pd( _mm_mul_pd(a.vec, b.vec), c.vec)}; }

// fmsub  dst = a*b-c
static inline vRealF SimdFmsub(const vRealF &a, const vRealF &b, const vRealF &c) { return {_mm_sub_ps(_mm_mul_ps(a.vec, b.vec), c.vec)}; }
static inline vRealD SimdFmsub(const vRealD &a, const vRealD &b, const vRealD &c) { return {_mm_sub_pd(_mm_mul_pd(a.vec, b.vec), c.vec)}; }

#endif

