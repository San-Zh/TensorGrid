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

#define _for(i, _N, e) \
    for (int i = 0; i < _N; i++) { e; }


template <typename Tp> struct vReal;
template <> struct vReal<float>  { float  vec[16]; enum{NumElem = 16}; };
template <> struct vReal<double> { double vec[8];  enum{NumElem = 8 }; };
// clang-format on

// tyepdef
typedef vReal<float>  vRealF;
typedef vReal<double> vRealD;

// load
template <typename Tp>
static inline vReal<Tp> SimdLoad(const Tp *_p)
{
    vReal<Tp> ret;
    _for(i, vReal<Tp>::NumElem, ret.vec[i] = _p[i]);
    return ret;
}

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

// add
template <typename Tp>
static inline void SimdSetzero(vReal<Tp> &a)
{
    _for(i, vReal<Tp>::NumElem, a.vec[i] = 0);
}

// add
template <typename Tp>
static inline vReal<Tp> SimdAdd(const vReal<Tp> &a, const vReal<Tp> b)
{
    vReal<Tp> ret;
    _for(i, vReal<Tp>::NumElem, ret.vec[i] = a.vec[i] + b.vec[i]);
    return ret;
}

// sub
template <typename Tp>
static inline vReal<Tp> SimdSub(const vReal<Tp> &a, const vReal<Tp> b)
{
    vReal<Tp> ret;
    _for(i, vReal<Tp>::NumElem, ret.vec[i] = a.vec[i] - b.vec[i]);
    return ret;
}

// mul
template <typename Tp>
static inline vReal<Tp> SimdMul(const vReal<Tp> &a, const vReal<Tp> b)
{
    vReal<Tp> ret;
    _for(i, vReal<Tp>::NumElem, ret.vec[i] = a.vec[i] * b.vec[i]);
    return ret;
}

// fmadd  dst = a*b+c
template <typename Tp>
static inline vReal<Tp> SimdFmadd(const vReal<Tp> &a, const vReal<Tp> b, const vReal<Tp> &c)
{
    vReal<Tp> ret;
    _for(i, vReal<Tp>::NumElem, ret.vec[i] = a.vec[i] * b.vec[i] + c.vec[i]);
    return ret;
}

// fmsub  dst = a*b-c
template <typename Tp>
static inline vReal<Tp> SimdFmsub(const vReal<Tp> &a, const vReal<Tp> b, const vReal<Tp> &c)
{
    vReal<Tp> ret;
    _for(i, vReal<Tp>::NumElem, ret.vec[i] = a.vec[i] * b.vec[i] - c.vec[i]);
    return ret;
}