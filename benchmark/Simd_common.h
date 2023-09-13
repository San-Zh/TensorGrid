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

#pragma once

// #include <iostream>

#define _for(i, _N, e) \
    for (int i = 0; i < _N; i++) { e; }


// clang-format off
template <typename Tp> struct vReal;
template <> struct vReal<float>  { float  vec[16]; enum{vlength = 16}; };
template <> struct vReal<double> { double vec[8];  enum{vlength = 8 }; };
// clang-format on

// tyepdef
typedef vReal<float>  vRealF;
typedef vReal<double> vRealD;

// load
template <typename Tp>
static inline vReal<Tp> SimdLoad(const Tp *_p)
{
    vReal<Tp> ret;
    _for(i, vReal<Tp>::vlength, ret.vec[i] = _p[i]);
    return ret;
}

// load
template <typename Tp>
static inline void SimdLoad(vReal<Tp> &a, const Tp *_p)
{
    _for(i, vReal<Tp>::vlength, a.vec[i] = _p[i]);
}

// store
template <typename Tp>
static inline void SimdStore(float *_p, const vReal<Tp> &a)
{
    _for(i, vReal<Tp>::vlength, _p[i] = a.vec[i]);
}

// add
template <typename Tp>
static inline vReal<Tp> SimdSetzero(const vReal<Tp> &a, const vReal<Tp> b)
{
    vReal<Tp> ret;
    _for(i, vReal<Tp>::vlength, ret.vec[i] = 0);
    return ret;
}

// add
template <typename Tp>
static inline vReal<Tp> SimdAdd(const vReal<Tp> &a, const vReal<Tp> b)
{
    vReal<Tp> ret;
    _for(i, vReal<Tp>::vlength, ret.vec[i] = a.vec[i] + b.vec[i]);
    return ret;
}

// sub
template <typename Tp>
static inline vReal<Tp> SimdSub(const vReal<Tp> &a, const vReal<Tp> b)
{
    vReal<Tp> ret;
    _for(i, vReal<Tp>::vlength, ret.vec[i] = a.vec[i] - b.vec[i]);
    return ret;
}

// mul
template <typename Tp>
static inline vReal<Tp> SimdMul(const vReal<Tp> &a, const vReal<Tp> b)
{
    vReal<Tp> ret;
    _for(i, vReal<Tp>::vlength, ret.vec[i] = a.vec[i] * b.vec[i]);
    return ret;
}

// fmadd  dst = a*b+c
template <typename Tp>
static inline vReal<Tp> SimdFmadd(const vReal<Tp> &a, const vReal<Tp> b, const vReal<Tp> &c)
{
    vReal<Tp> ret;
    _for(i, vReal<Tp>::vlength, ret.vec[i] = a.vec[i] * b.vec[i] + c.vec[i]);
    return ret;
}

// fmsub  dst = a*b-c
template <typename Tp>
static inline vReal<Tp> SimdFmsub(const vReal<Tp> &a, const vReal<Tp> b, const vReal<Tp> &c)
{
    vReal<Tp> ret;
    _for(i, vReal<Tp>::vlength, ret.vec[i] = a.vec[i] * b.vec[i] - c.vec[i]);
    return ret;
}