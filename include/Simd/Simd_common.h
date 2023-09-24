/**
 * @file Simd_common.h
 * @author your name (you@domain.com)
 * @brief Real<float>, vReal<double> for a \b virtual 512-bit simd vector. 
 * @version 0.1
 * @date 2023-09-09
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#pragma message(" TensorGrid/include/Simd/Simd_common.h included")

// #include <iostream>
// #include <string.h>
#include <type_traits>

#define _for(i, _N, e) \
    for (int i = 0; i < _N; i++) { e; }

// clang-format off

/**
 * @brief vReal<float>, vReal<double> for a \b virtual 512-bit simd vector. The purpose is to 
 * achieve automatic vectorization through this virtual vector behaves like a real one.
 * \todo NOT WELL EFFECTS. 
 * 
 * @tparam Tp 
 */
template<typename Tp> struct _SelfSimdWidthTraits;
template<> struct _SelfSimdWidthTraits<double> { enum {_wd = 8}; };
template<> struct _SelfSimdWidthTraits<float> { enum {_wd = 16}; };


template <typename Tp>
struct vReal {
    float vec[_SelfSimdWidthTraits<Tp>::_wd];
    enum { NumElem = _SelfSimdWidthTraits<Tp>::_wd };

    inline void load(const Tp *_p, const size_t _ofs) { _for(_i, vReal<Tp>::NumElem, vec[_i] = _p[_ofs + _i]); }
    inline void load(const Tp *_p) { _for(_i, vReal<Tp>::NumElem, vec[_i] = _p[_i]); }

    inline void store(Tp *_p, const size_t _ofs) { _for(_i, vReal<Tp>::NumElem, _p[_ofs +_i] = vec[_i]); }
    inline void store(Tp *_p) { _for(_i, vReal<Tp>::NumElem, _p[_i] = vec[_i]); }

    inline void setzero() { _for(_i, vReal<Tp>::NumElem, vec[_i] = 0.0); }
    inline void set(const Tp &_a) {  _for(_i, vReal<Tp>::NumElem, vec[_i] = _a); }
};


// clang-format on

// tyepdef
typedef vReal<float>  vRealF;
typedef vReal<double> vRealD;


///////////////// with a void type return; ////////////////////

// load
template <typename Tp>
static inline void SimdLoad(vReal<Tp> &a, const Tp *_p)
{
    _for(i, vReal<Tp>::NumElem, a.vec[i] = _p[i]);
}

// store
template <typename Tp>
static inline void SimdStore(Tp *_p, const vReal<Tp> &a)
{
    _for(i, vReal<Tp>::NumElem, _p[i] = a.vec[i]);
}

// set zero
template <typename Tp>
static inline void SimdSetzero(vReal<Tp> &a)
{
    _for(i, vReal<Tp>::NumElem, a.vec[i] = 0);
}

// add
template <typename Tp>
static inline void SimdAdd(vReal<Tp> &ret, const vReal<Tp> &a, const vReal<Tp> &b)
{
    _for(i, vReal<Tp>::NumElem, ret.vec[i] = a.vec[i] + b.vec[i]);
}

// sub
template <typename Tp>
static inline void SimdSub(vReal<Tp> &ret, const vReal<Tp> &a, const vReal<Tp> &b)
{
    _for(i, vReal<Tp>::NumElem, ret.vec[i] = a.vec[i] - b.vec[i]);
}

// mul
template <typename Tp>
static inline void SimdMul(vReal<Tp> &ret, const vReal<Tp> &a, const vReal<Tp> &b)
{
    _for(i, vReal<Tp>::NumElem, ret.vec[i] = a.vec[i] * b.vec[i]);
}

// fmadd  dst = a*b+c
template <typename Tp>
static inline void SimdFmadd(vReal<Tp> &ret, const vReal<Tp> &a, const vReal<Tp> &b,
                             const vReal<Tp> &c)
{
    _for(i, vReal<Tp>::NumElem, ret.vec[i] = a.vec[i] * b.vec[i] + c.vec[i]);
}

// fmsub  dst = a*b-c
template <typename Tp>
static inline void SimdFmsub(vReal<Tp> &ret, const vReal<Tp> &a, const vReal<Tp> &b,
                             const vReal<Tp> &c)
{
    _for(i, vReal<Tp>::NumElem, ret.vec[i] = a.vec[i] * b.vec[i] - c.vec[i]);
}


//////////////////// with a vReal<Tp> type return; ///////////////////

// load
template <typename Tp>
static inline vReal<Tp> SimdLoad(const Tp *_p)
{
    vReal<Tp> ret;
    _for(i, vReal<Tp>::NumElem, ret.vec[i] = _p[i]);
    return ret;
}

// add
template <typename Tp>
static inline vReal<Tp> SimdAdd(const vReal<Tp> &a, const vReal<Tp> &b)
{
    vReal<Tp> ret;
    _for(i, vReal<Tp>::NumElem, ret.vec[i] = a.vec[i] + b.vec[i]);
    return ret;
}

// sub
template <typename Tp>
static inline vReal<Tp> SimdSub(const vReal<Tp> &a, const vReal<Tp> &b)
{
    vReal<Tp> ret;
    _for(i, vReal<Tp>::NumElem, ret.vec[i] = a.vec[i] - b.vec[i]);
    return ret;
}

// mul
template <typename Tp>
static inline vReal<Tp> SimdMul(const vReal<Tp> &a, const vReal<Tp> &b)
{
    vReal<Tp> ret;
    _for(i, vReal<Tp>::NumElem, ret.vec[i] = a.vec[i] * b.vec[i]);
    return ret;
}

// fmadd  dst = a*b+c
template <typename Tp>
static inline vReal<Tp> SimdFmadd(const vReal<Tp> &a, const vReal<Tp> &b, const vReal<Tp> &c)
{
    vReal<Tp> ret;
    _for(i, vReal<Tp>::NumElem, ret.vec[i] = a.vec[i] * b.vec[i] + c.vec[i]);
    return ret;
}

// fmsub  dst = a*b-c
template <typename Tp>
static inline vReal<Tp> SimdFmsub(const vReal<Tp> &a, const vReal<Tp> &b, const vReal<Tp> &c)
{
    vReal<Tp> ret;
    _for(i, vReal<Tp>::NumElem, ret.vec[i] = a.vec[i] * b.vec[i] - c.vec[i]);
    return ret;
}