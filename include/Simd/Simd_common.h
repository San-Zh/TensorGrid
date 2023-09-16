/**
 * @file Simd_common.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-09
 * 
 * @copyright Copyright (c) 2023
 * 
 */

// clang-format off

#pragma once

#pragma message(" \"Simd_common.h\" included")

// #include <iostream>
#include <string.h>

#define _for(i, _N, e) \
    for (int i = 0; i < _N; i++) { e; }

constexpr unsigned W_BITS = 512;
constexpr unsigned W_BYTES = W_BITS >> 3;
constexpr unsigned W_PS = W_BITS >>5 ;
constexpr unsigned W_PD = W_BITS >>6;

template <typename Tp> struct vReal;
template <> struct vReal<float>  { float  vec[W_PS]; enum{NumElem = W_PS}; };
template <> struct vReal<double> { double vec[W_PD]; enum{NumElem = W_PD }; };
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
    // memcpy(a.vec, _p, W_BYTES);
}

// store
template <typename Tp>
static inline void SimdStore(Tp *_p, const vReal<Tp> &a)
{
    _for(i, vReal<Tp>::NumElem, _p[i] = a.vec[i]);
    // memcpy(_p, a.vec, W_BYTES);
}

// set zero
template <typename Tp>
static inline void SimdSetzero(vReal<Tp> &a)
{
    _for(i, vReal<Tp>::NumElem, a.vec[i] = 0);
    // memset(a.vec, 0, W_BYTES);
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
    // memcpy(ret.vec, _p, 64);
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