/**
 * @file Simd_complex.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-09
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include "Simd_opt.h"

/**
 * @brief 
 * 
 * @tparam Tp represent presicion. Now float(F32) and double(F64) is supported
 * \todo TODO word std::complex<Tp> is    
 */
template <typename Tp>
struct vComplex {
    vReal<Tp> real;
    vReal<Tp> imag;
    enum { NumElem = vReal<Tp>::NumElem };
    /// Define arithmetic operators: FIX IT
    // friend inline vComplex<Tp> operator+(vComplexD<Tp>a, vComplex<Tp> b);
    // friend inline vComplex<Tp> operator-(vComplexD<Tp>a, vComplex<Tp> b);
    // friend inline vComplex<Tp> operator*(vComplexD<Tp>a, vComplex<Tp> b);
    // friend inline vComplex<Tp> operator/(vComplexD<Tp>a, vComplex<Tp> b);
};


// clang-format off

using vComplexF = vComplex<float >;
using vComplexD = vComplex<double>;


template <typename Tp> struct vComplexTraits;
template <> struct vComplexTraits<float > { typedef vComplex<float > value_type; };
template <> struct vComplexTraits<double> { typedef vComplex<double> value_type; };

// clang-format on

///////////////// with a void type return; ////////////////////

// load
template <typename Tp>
static inline void SimdLoad(vComplex<Tp> &a, const Tp *p_re, const Tp *p_im)
{
    SimdLoad(a.real, p_re);
    SimdLoad(a.imag, p_im);
}

template <typename Tp>
static inline void SimdLoad(vComplex<Tp> &a, Tp *p[2])
{
    SimdLoad(a.real, p[0]);
    SimdLoad(a.imag, p[1]);
}


// store
template <typename Tp>
static inline void SimdStore(Tp *p_re, Tp *p_im, const vComplex<Tp> &a)
{
    SimdStore(p_re, a.real);
    SimdStore(p_im, a.imag);
}

template <typename Tp>
static inline void SimdStore(Tp *_p[2], const vComplex<Tp> &a)
{
    SimdStore(_p[0], a.real);
    SimdStore(_p[1], a.imag);
}

// set zero
template <typename Tp>
static inline void SimdSetzero(vComplex<Tp> &ret)
{
    SimdSetzero(ret.real), SimdSetzero(ret.imag);
}

// add
template <typename Tp>
static inline void SimdAdd(vComplex<Tp> &ret, const vComplex<Tp> &a, const vComplex<Tp> &b)
{
    SimdAdd(ret.real, a.real, b.real);
    SimdAdd(ret.imag, a.imag, b.imag);
}

// sub
template <typename Tp>
static inline void SimdSub(vComplex<Tp> &ret, const vComplex<Tp> &a, const vComplex<Tp> &b)
{
    SimdSub(ret.real, a.real, b.real);
    SimdSub(ret.imag, a.imag, b.imag);
}

// mul
template <typename Tp>
static inline void SimdMul(vComplex<Tp> &ret, const vComplex<Tp> &a, const vComplex<Tp> &b)
{
    SimdFmsub(a.real, b.real, SimdMul(a.imag, b.imag));
    SimdFmadd(a.real, b.imag, SimdMul(a.imag, b.real));
}

// fmadd
template <typename Tp>
static inline void SimdFmadd(vComplex<Tp> &ret, const vComplex<Tp> &a, const vComplex<Tp> &b,
                             const vComplex<Tp> &c)
{
    SimdFmsub(ret.real, a.imag, b.imag, c.real);
    SimdFmsub(ret.real, a.real, b.real, ret.real);
    SimdFmadd(ret.imag, a.imag, b.real, c.imag);
    SimdFmadd(ret.imag, a.real, b.imag, ret.imag);
}

// fmsub
template <typename Tp>
static inline void SimdFmsub(vComplex<Tp> &ret, const vComplex<Tp> &a, const vComplex<Tp> &b,
                             const vComplex<Tp> &c)
{
    SimdFmadd(ret.real, a.imag, b.imag, c.real);
    SimdFmsub(ret.real, a.real, b.real, ret.real);
    SimdFmsub(ret.imag, a.imag, b.real, c.imag);
    SimdFmadd(ret.imag, a.real, b.imag, ret.imag);
}


//////////////////// with a vComplex<Tp> type return; ///////////////////

// load
template <typename Tp>
static inline vComplex<Tp> SimdLoad(const Tp *p_re, const Tp *p_im)
{
    return {SimdLoad(p_re), SimdLoad(p_im)};
}

// add
template <typename Tp>
static inline vComplex<Tp> SimdAdd(const vComplex<Tp> &a, const vComplex<Tp> &b)
{
    return {SimdAdd(a.real, b.real), SimdAdd(a.imag, b.imag)};
}

// sub
template <typename Tp>
static inline vComplex<Tp> SimdSub(const vComplex<Tp> &a, const vComplex<Tp> &b)
{
    return {SimdSub(a.real, b.real), SimdSub(a.imag, b.imag)};
}

// mul
template <typename Tp>
static inline vComplex<Tp> SimdMul(const vComplex<Tp> &a, const vComplex<Tp> &b)
{
    return {SimdFmsub(a.real, b.real, SimdMul(a.imag, b.imag)),
            SimdFmsub(a.real, b.imag, SimdMul(a.imag, b.real))};
}

// fmadd  dst = a*b + c; dst_r = ar * br - ai * bi + cr = (ar * br + cr) - ai * bi
template <typename Tp>
static inline vComplex<Tp> SimdFmadd(const vComplex<Tp> &a, const vComplex<Tp> &b,
                                     const vComplex<Tp> &c)
{
    return {SimdFmsub(a.real, b.real, SimdFmsub(a.imag, b.imag, c.real)),
            SimdFmadd(a.real, b.imag, SimdFmadd(a.imag, b.real, c.imag))};
}

// fmsub  dst = a*b - c; dst_r = ar * br - ai * bi + cr = (ar * br + cr) - ai * bi
template <typename Tp>
static inline vComplex<Tp> SimdFmsub(const vComplex<Tp> &a, const vComplex<Tp> &b,
                                     const vComplex<Tp> &c)
{
    return {SimdFmsub(a.real, b.real, SimdFmadd(a.imag, b.imag, c.real)),
            SimdFmadd(a.real, b.imag, SimdFmsub(a.imag, b.real, c.imag))};
}



//////////////////
#if 0

template <>
template <typename Tp>
struct vComplex<std::complex<Tp>> {
    vReal<Tp> real;
    vReal<Tp> imag;
};

void A() { vComplex<std::complex<float>> a; a.real.vec.zero();}



// struct vComplexF { vRealF real; vRealF imag; };
// struct vComplexD { vRealD real; vRealD imag; };

#endif