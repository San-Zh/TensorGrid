/**
 * @file SimdTupleKernal.h
 * @author your name (you@domain.com)
 * @brief Simd Kernals, like aXpY, XdotY, XoYpM
 * @version 0.1
 * @date 2023-09-23
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include "SimdTupleTypes.h"


/**
 * @brief Y[n] = va * X[n] + Y[n],   (n=0,...,_N-1)
 * 
 * @tparam vtype 
 * @tparam _N 
 * @param va 
 * @param X 
 * @param Y 
 */
template <class vtype, unsigned _N>
static inline void kernal_simd_aXpY(const vtype &va, const iVector<vtype, _N> &X,
                                    iVector<vtype, _N> &Y)
{
    for (unsigned n = 0; n < _N; n++) { Y(n) = SimdFmadd(va, X(n), Y(n)); }
}


/**
 * @brief iVerctor OUTER PRODUCT: Mat[m][n] = X[m] * Y[n] + M[m][n], ie. Mat = X \\otimes Y + M 
 * 
 * @tparam vtype 
 * @tparam _M 
 * @tparam _N 
 * @param Mat iMatrix<vtype, _M, _N>
 * @param X iVector<vtype, _M>
 * @param Y iVector<vtype, _M>
 */
template <class vtype, unsigned _M, unsigned _N>
static inline void kernal_simd_XoYpM(iMatrix<vtype, _M, _N> &Mat, const iVector<vtype, _M> &X,
                                     const iVector<vtype, _N> &Y)
{
    for (unsigned m = 0; m < _M; m++) {
        for (unsigned n = 0; n < _N; n++) {
            Mat(m, n) = SimdFmadd(X(m), Y(n), Mat(m, n));
            // SimdFmadd(Mat(m, n), X(m), Y(n), Mat(m, n));
        }
    }
}


/**
 * @brief iVerctor INNER PRODUCT: return $vtype ret = \\sum_n X[n] * Y[n]$
 * 
 * @tparam vtype 
 * @tparam _N 
 * @param X 
 * @param Y 
 * @return vtype 
 */
template <class vtype, unsigned _N>
static inline void kernal_simd_XdotY(vtype &ret, const iVector<vtype, _N> &X,
                                     const iVector<vtype, _N> &Y)
{
    for (unsigned n = 0; n < _N; n++) { ret = SimdFmadd(X(n), Y(n), ret); }
}


/**
 * @brief 
 * 
 * @tparam vtype 
 * @tparam _M 
 * @tparam _N 
 * @tparam _K 
 * @param C 
 * @param A 
 * @param B 
 */
template <class vtype, unsigned _M, unsigned _N, unsigned _K>
static inline void kernal_simd_gemm(iMatrix<vtype, _M, _N> &C, const iMatrix<vtype, _M, _K> &A,
                                    const iMatrix<vtype, _K, _N> &B)
{
    for (unsigned k = 0; k < _K; k++) {
        for (unsigned m = 0; m < _M; m++) {
            for (unsigned n = 0; n < _N; n++) { SimdFmadd(C(m, n), A(m, k), B(k, n), C(m, n)); }
        }
    }
}