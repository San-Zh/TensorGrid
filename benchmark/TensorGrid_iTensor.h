/**
 * @file SimdTypes.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-10
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include "Simd_complex.h"

// clang-format off


template <typename vtype> struct SimdTraits;
// template <> struct SimdTraits<vRealF>    { typedef vRealF    value_type; }; // enum{ num_value = sizeof(vRealF)};
// template <> struct SimdTraits<vRealD>    { typedef vRealD    value_type; }; // enum{ num_value = sizeof(vRealD)};
template <> struct SimdTraits<vComplexF> { typedef vComplexF value_type; }; // enum{ num_value = sizeof(vRealF)};
template <> struct SimdTraits<vComplexD> { typedef vComplexD value_type; }; // enum{ num_value = sizeof(vRealD)};
// clang-format on

template <class vtype, int N>
class iVector {
  public:
    typename SimdTraits<vtype>::value_type vec[N];
};

template <class vtype, int M, int N = M>
class iMatrix {
  public:
    typename SimdTraits<vtype>::value_type mat[M][N];
};

/**
 * @brief 
 * 
 * @tparam vtype 
 * @tparam M 
 * @tparam N 
 * @param VA 
 * @param VB 
 * @param MC 
 */
template <typename vtype, int M, int N>
void simdKernal_TXYpM(iVector<vtype, M> &VA, iVector<vtype, N> &VB, iMatrix<vtype, M, N> &MC)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            MC.mat[m][n] = SimdFmadd(VA.vec[m], VB.vec[n], MC.mat[m][n]);
        }
    }
}

/**
 * @brief C[m][n] = \sum_k A[m][k] * B[k][n] + C[m][n]
 * 
 * @tparam vtype 
 * @tparam M 
 * @tparam N 
 * @tparam K 
 * @param A 
 * @param B 
 * @param C 
 * @return auto 
 */
template <typename vtype, int M, int N, int K>
void simdkernal_gemm(iMatrix<vtype, M, K> &A, iMatrix<vtype, K, N> &B, iMatrix<vtype, M, N> &C)
{
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                C.mat[m][n] = SimdFmadd(A.vec[m], B.vec[n], C.mat[m][n]);
            }
        }
    }
}