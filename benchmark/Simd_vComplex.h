/**
 * @file Simd_vComplex.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-09
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "Simd_common.h"
#include "Simd_avx512.h"
// #include "Simd_avx.h"
// #include "Simd_sve.h"

struct vComplexF {
    vRealF real;
    vRealF imag;
};

struct vComplexD {
    vRealD real;
    vRealD imag;
    // Define arithmetic operators
    // friend inline vComplexD operator+(vComplexD a, vComplexD b);
    // friend inline vComplexD operator-(vComplexD a, vComplexD b);
    // friend inline vComplexD operator*(vComplexD a, vComplexD b);
    // friend inline vComplexD operator/(vComplexD a, vComplexD b);
};

// load
static inline vComplexF SimdLoad(const float *p_re, const float *p_im) { return {SimdLoad(p_re), SimdLoad(p_im)}; }
static inline vComplexD SimdLoad(const double *p_re, const double *p_im) { return {SimdLoad(p_re), SimdLoad(p_im)}; }

// store
static inline void vStore(float *p_re, float *p_im, vComplexF a)
{
    SimdStore(p_re, a.real);
    SimdStore(p_im, a.imag);
}
static inline void vStore(double *p_re, double *p_im, vComplexD a)
{
    SimdStore(p_re, a.real);
    SimdStore(p_im, a.imag);
}

// complex mul
// {SimdSub(SimdMul(a.real, b.real), SimdMul(a.imag, b.imag)), SimdAdd(SimdMul(a.real, b.imag), SimdMul(a.imag, b.real))};
static inline vComplexF SimdMul(const vComplexF &a, const vComplexF &b)
{
    return {SimdFmsub(a.real, b.real, SimdMul(a.imag, b.imag)), SimdFmadd(a.real, b.imag, SimdMul(a.imag, b.real))};
}

static inline vComplexD SimdMul(const vComplexD &a, const vComplexD &b)
{
    return {SimdFmsub(a.real, b.real, SimdMul(a.imag, b.imag)), SimdFmadd(a.real, b.imag, SimdMul(a.imag, b.real))};
}

// complex fma  dst = a*b + c; dst_r = ar * br - ai * bi + cr = (ar * br + cr) - ai * bi
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
