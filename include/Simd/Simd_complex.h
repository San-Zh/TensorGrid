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

// load
template <typename Tp>
static inline vComplex<Tp> SimdLoad(const Tp *p_re, const Tp *p_im)
{
    return {SimdLoad(p_re), SimdLoad(p_im)};
}

template <typename Tp>
static inline void SimdLoad(vComplex<Tp> &a, const Tp *p_re, const Tp *p_im)
{
    SimdLoad(a.real, p_re);
    SimdLoad(a.imag, p_im);
}

// store
template <typename Tp>
static inline void SimdStore(Tp *p_re, Tp *p_im, const vComplex<Tp> &a)
{
    SimdStore(p_re, a.real);
    SimdStore(p_im, a.imag);
}

// set zero
template <typename Tp>
static inline void SimdSetzero(vComplex<Tp> &a)
{
    SimdSetzero(a.real), SimdSetzero(a.imag);
}
// static inline void SimdSetzero(vComplexD &) { SimdSetzero(a.real), SimdSetzero(a.imag); }


// add
static inline vComplexF SimdAdd(const vComplexF &a, const vComplexF &b)
{
    return {SimdAdd(a.real, b.real), SimdAdd(a.imag, b.imag)};
}

static inline vComplexD SimdAdd(const vComplexD &a, const vComplexD &b)
{
    return {SimdAdd(a.real, b.real), SimdAdd(a.imag, b.imag)};
}


// mul
static inline vComplexF SimdMul(const vComplexF &a, const vComplexF &b)
{
    return {SimdFmsub(a.real, b.real, SimdMul(a.imag, b.imag)),
            SimdFmadd(a.real, b.imag, SimdMul(a.imag, b.real))};
}

static inline vComplexD SimdMul(const vComplexD &a, const vComplexD &b)
{
    return {SimdFmsub(a.real, b.real, SimdMul(a.imag, b.imag)),
            SimdFmadd(a.real, b.imag, SimdMul(a.imag, b.real))};
}


// fma  dst = a*b + c; dst_r = ar * br - ai * bi + cr = (ar * br + cr) - ai * bi
static inline vComplexF SimdFmadd(const vComplexF &a, const vComplexF &b, const vComplexF &c)
{
    return {SimdFmsub(a.real, b.real, SimdFmsub(a.imag, b.imag, c.real)),
            SimdFmadd(a.real, b.imag, SimdFmadd(a.imag, b.real, c.imag))};
}

static inline vComplexD SimdFmadd(const vComplexD &a, const vComplexD &b, const vComplexD &c)
{
    return {SimdFmsub(a.real, b.real, SimdFmsub(a.imag, b.imag, c.real)),
            SimdFmadd(a.real, b.imag, SimdFmadd(a.imag, b.real, c.imag))};
}

// fma  dst = a*b - c; dst_r = ar * br - ai * bi + cr = (ar * br + cr) - ai * bi
static inline vComplexF SimdFmsub(const vComplexF &a, const vComplexF &b, const vComplexF &c)
{
    return {SimdFmsub(a.real, b.real, SimdFmsub(a.imag, b.imag, c.real)),
            SimdFmadd(a.real, b.imag, SimdFmadd(a.imag, b.real, c.imag))};
}

static inline vComplexD SimdFmasub(const vComplexD &a, const vComplexD &b, const vComplexD &c)
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