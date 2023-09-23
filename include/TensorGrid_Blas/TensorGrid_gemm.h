/**
 * @file TensorGrid_gemm.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-20
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include "Simd/SimdTupleKernal.h"



/**
 * @brief v0.0.0: Gemm  C[m][n] = sum_k A[m][k] * B[k][n];
 * 
 * @tparam M 
 * @tparam N 
 * @tparam K 
 * @tparam Tp 
 * @param A 
 * @param B 
 * @param C 
 * @param gridSize 
 */
template <unsigned M, unsigned N, unsigned K, typename Tp>
void TensorGrid_complex_gemm(const Tp *A, const Tp *B, Tp *C, size_t gridSize)
{
    typedef typename SimdTraits<vComplex<Tp>>::ptr_type CTp;

    // CTp pA[M][K];
    CTp pA[K][M];
    CTp pB[K][N];
    CTp pC[M][N];
    for (unsigned m = 0; m < M; m++) {
        for (unsigned k = 0; k < K; k++) {
            pA[k][m][0] = const_cast<Tp *>(&A[(m * K * 2 + k * 2 + 0) * gridSize]);
            pA[k][m][1] = const_cast<Tp *>(&A[(m * K * 2 + k * 2 + 1) * gridSize]);
        }
    }
    for (unsigned k = 0; k < K; k++) {
        for (unsigned n = 0; n < N; n++) {
            pB[k][n][0] = const_cast<Tp *>(&B[(k * N * 2 + n * 2 + 0) * gridSize]);
            pB[k][n][1] = const_cast<Tp *>(&B[(k * N * 2 + n * 2 + 1) * gridSize]);
        }
    }
    for (unsigned m = 0; m < M; m++) {
        for (unsigned n = 0; n < N; n++) {
            pC[m][n][0] = const_cast<Tp *>(&C[(m * N * 2 + n * 2 + 0) * gridSize]);
            pC[m][n][1] = const_cast<Tp *>(&C[(m * N * 2 + n * 2 + 1) * gridSize]);
        }
    }

    iMatrix<vComplex<Tp>, M, N> Mc;
    iVector<vComplex<Tp>, M>    Va;
    iVector<vComplex<Tp>, N>    Vb;
    for (size_t v = 0; v < gridSize; v += vComplex<Tp>::NumElem) {
        Mc.setzero();
        for (unsigned k = 0; k < K; k++) {
            for (unsigned m = 0; m < M; m++) {
                Va(m).load(pA[k][m], v);
                for (unsigned n = 0; n < N; n++) {
                    Vb(n).load(pB[k][n], v);
                    Mc(m, n) = SimdFmadd(Va(m), Vb(n), Mc(m, n));
                }
            }
        }
        Mc.store(pC, v);
    }
}



/**
 * @brief v0.0.1: Gemm  C[m][n] = sum_k A[m][k] * B[k][n];
 * 
 * @tparam M 
 * @tparam N 
 * @tparam K 
 * @tparam Tp 
 * @param A 
 * @param B 
 * @param C 
 * @param gridSize 
 */
template <unsigned M, unsigned N, unsigned K, typename Tp>
void TensorGrid_complex_gemm_v1(const Tp *A, const Tp *B, Tp *C, size_t gridSize)
{
    typedef typename SimdTraits<vComplex<Tp>>::ptr_type CTp;

    // CTp pA[M][K];
    CTp pA[K][M];
    CTp pB[K][N];
    CTp pC[M][N];
    for (unsigned m = 0; m < M; m++) {
        for (unsigned k = 0; k < K; k++) {
            pA[k][m][0] = const_cast<Tp *>(&A[(m * K * 2 + k * 2 + 0) * gridSize]);
            pA[k][m][1] = const_cast<Tp *>(&A[(m * K * 2 + k * 2 + 1) * gridSize]);
        }
    }
    for (unsigned k = 0; k < K; k++) {
        for (unsigned n = 0; n < N; n++) {
            pB[k][n][0] = const_cast<Tp *>(&B[(k * N * 2 + n * 2 + 0) * gridSize]);
            pB[k][n][1] = const_cast<Tp *>(&B[(k * N * 2 + n * 2 + 1) * gridSize]);
        }
    }
    for (unsigned m = 0; m < M; m++) {
        for (unsigned n = 0; n < N; n++) {
            pC[m][n][0] = const_cast<Tp *>(&C[(m * N * 2 + n * 2 + 0) * gridSize]);
            pC[m][n][1] = const_cast<Tp *>(&C[(m * N * 2 + n * 2 + 1) * gridSize]);
        }
    }

    for (size_t v = 0; v < gridSize; v += vComplex<Tp>::NumElem) {
        iMatrix<vComplex<Tp>, M, N> Mc;
        Mc.setzero();
        for (unsigned k = 0; k < K; k++) {
            iVector<vComplex<Tp>, M> Va;
            Va.load(pA[k], v);
            iVector<vComplex<Tp>, N> Vb;
            Vb.load(pB[k], v);
            kernal_simd_XoYpM(Mc, Va, Vb);
        }
        Mc.store(pC, v);
    }
}



/**
 * @brief v0.0.0: Gemm  C[m][n] = sum_k A[m][k] * B[k][n];
 * 
 * @tparam M 
 * @tparam N 
 * @tparam K 
 * @tparam Tp 
 * @param A 
 * @param B 
 * @param C 
 * @param gridSize 
 */
template <unsigned M, unsigned N, unsigned K, typename Tp>
void TensorGrid_real_gemm(const Tp *A, const Tp *B, Tp *C, size_t gridSize)
{
    typedef typename SimdTraits<vReal<Tp>>::ptr_type CTp;

    // CTp pA[M][K];
    CTp pA[K][M];
    CTp pB[K][N];
    CTp pC[M][N];
    for (unsigned m = 0; m < M; m++) {
        for (unsigned k = 0; k < K; k++) {
            pA[k][m] = const_cast<Tp *>(&A[(m * K + k) * gridSize]);
        }
    }
    for (unsigned k = 0; k < K; k++) {
        for (unsigned n = 0; n < N; n++) {
            pB[k][n] = const_cast<Tp *>(&B[(k * N + n) * gridSize]);
        }
    }
    for (unsigned m = 0; m < M; m++) {
        for (unsigned n = 0; n < N; n++) {
            pC[m][n] = const_cast<Tp *>(&C[(m * N + n) * gridSize]);
        }
    }

    iMatrix<vReal<Tp>, M, N> Mc;
    iVector<vReal<Tp>, M>    Va;
    iVector<vReal<Tp>, N>    Vb;
    for (size_t v = 0; v < gridSize; v += vReal<Tp>::NumElem) {
        Mc.setzero();
        for (unsigned k = 0; k < K; k++) {
            for (unsigned m = 0; m < M; m++) {
                Va(m).load(pA[k][m], v);
                for (unsigned n = 0; n < N; n++) {
                    Vb(n).load(pB[k][n], v);
                    Mc(m, n) = SimdFmadd(Va(m), Vb(n), Mc(m, n));
                }
            }
        }
        Mc.store(pC, v);
    }
}



/**
 * @brief v0.0.1: Gemm  C[m][n] = sum_k A[m][k] * B[k][n];
 * 
 * @tparam M 
 * @tparam N 
 * @tparam K 
 * @tparam Tp 
 * @param A 
 * @param B 
 * @param C 
 * @param gridSize 
 */
template <unsigned M, unsigned N, unsigned K, typename Tp>
void TensorGrid_real_gemm_v1(const Tp *A, const Tp *B, Tp *C, size_t gridSize)
{
    typedef typename SimdTraits<vReal<Tp>>::ptr_type CTp;

    // CTp pA[M][K];
    CTp pA[K][M];
    CTp pB[K][N];
    CTp pC[M][N];
    for (unsigned m = 0; m < M; m++) {
        for (unsigned k = 0; k < K; k++) {
            pA[k][m] = const_cast<Tp *>(&A[(m * K + k) * gridSize]);
        }
    }
    for (unsigned k = 0; k < K; k++) {
        for (unsigned n = 0; n < N; n++) {
            pB[k][n] = const_cast<Tp *>(&B[(k * N + n) * gridSize]);
        }
    }
    for (unsigned m = 0; m < M; m++) {
        for (unsigned n = 0; n < N; n++) {
            pC[m][n] = const_cast<Tp *>(&C[(m * N + n) * gridSize]);
        }
    }

    for (size_t v = 0; v < gridSize; v += vReal<Tp>::NumElem) {
        iMatrix<vReal<Tp>, M, N> Mc;
        Mc.setzero();
        for (unsigned k = 0; k < K; k++) {
            iVector<vReal<Tp>, M> Va;
            Va.load(pA[k], v);
            iVector<vReal<Tp>, N> Vb;
            Vb.load(pB[k], v);
            kernal_simd_XoYpM(Mc, Va, Vb);
        }
        Mc.store(pC, v);
    }
}