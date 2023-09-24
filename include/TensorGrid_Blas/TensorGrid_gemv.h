/**
 * @file TensorGrid_gemv.h
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

#define version

/**
 * @brief v0.0.0: TensorGrid_complex_gemv
 * 
 * @tparam M 
 * @tparam N 
 * @tparam Tp 
 * @param A 
 * @param X 
 * @param Y 
 * @param gridSize 
 */
template <unsigned M, unsigned N, typename Tp>
void TensorGrid_complex_gemv(const Tp *A, const Tp *X, Tp *Y, size_t gridSize)
{
    Tp *Mp[M][N][2];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            Mp[i][j][0] = const_cast<Tp *>(&A[(i * N * 2 + j * 2 + 0) * gridSize]);
            Mp[i][j][1] = const_cast<Tp *>(&A[(i * N * 2 + j * 2 + 1) * gridSize]);
        }
    }
    Tp *Vdp[M][2];
    for (int i = 0; i < M; i++) {
        Vdp[i][0] = &Y[(i * 2 + 0) * gridSize];
        Vdp[i][1] = &Y[(i * 2 + 1) * gridSize];
    }
    Tp *Vsp[N][2];
    for (int j = 0; j < N; j++) {
        Vsp[j][0] = const_cast<Tp *>(&X[(j * 2 + 0) * gridSize]);
        Vsp[j][1] = const_cast<Tp *>(&X[(j * 2 + 1) * gridSize]);
    }

    // Vd[0] += M[0][n] * Vs[0]
    // Vd[:] += M[:][n] * Vs[:]
    // Vd[m] += M[m][n] * Vs[m]
    // iVector<vComplex<Tp>, M> Vd;
    // iVector<vComplex<Tp>, M> Mn;
    vComplex<Tp> Vd[M];
    vComplex<Tp> Mn[M];
    vComplex<Tp> vs;
    for (size_t v = 0; v < gridSize; v += vComplex<Tp>::NumElem) {
        for (size_t m = 0; m < M; m++) { SimdSetzero(Vd[m]); }
        for (size_t n = 0; n < N; n++) {
            vs.load(&Vsp[n][0][v], &Vsp[n][1][v]); ///
            for (size_t m = 0; m < M; m++) {
                Mn[m].load(&Mp[m][n][0][v], &Mp[m][n][1][v]); ///
                Vd[m] = SimdFmadd(Mn[m], vs, Vd[m]);
                // SimdLoad(Mn(m), &Mp[m][n][0][v], &Mp[m][n][1][v]); ///
                // SimdFmadd(Vd(m), Mn(m), vs, Vd(m));
            }
        }

        for (int m = 0; m < M; m++) { Vd[m].store(&Vdp[m][0][v], &Vdp[m][1][v]); }
    }
}




/**
 * @brief v0.0.1 : outer product method
 * 
 * @tparam M 
 * @tparam N 
 * @tparam Tp 
 * @param A 
 * @param X 
 * @param Y 
 * @param gridSize 
 */
template <unsigned M, unsigned N, typename Tp>
void TensorGrid_complex_gemv_v1(const Tp *A, const Tp *X, Tp *Y, const size_t gridSize)
{
    typedef typename SimdTraits<vComplex<Tp>>::ptr_type CTp;

    CTp Mm_p[N][M];
    CTp Vd_p[M];
    CTp Vs_p[N];

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            Mm_p[j][i][0] = const_cast<Tp *>(&A[(i * N * 2 + j * 2 + 0) * gridSize]);
            Mm_p[j][i][1] = const_cast<Tp *>(&A[(i * N * 2 + j * 2 + 1) * gridSize]);
        }
    }
    for (int i = 0; i < M; i++) {
        Vd_p[i][0] = &Y[(i * 2 + 0) * gridSize];
        Vd_p[i][1] = &Y[(i * 2 + 1) * gridSize];
    }
    for (int j = 0; j < N; j++) {
        Vs_p[j][0] = const_cast<Tp *>(&X[(j * 2 + 0) * gridSize]);
        Vs_p[j][1] = const_cast<Tp *>(&X[(j * 2 + 1) * gridSize]);
    }

    for (size_t v = 0; v < gridSize; v += vComplex<Tp>::NumElem) {
        iVector<vComplex<Tp>, M> Vd;
        Vd.setzero();
        for (size_t n = 0; n < N; n++) {
            vComplex<Tp> vs;
            vs.load(Vs_p[n], v);
            iVector<vComplex<Tp>, M> Vm;
            Vm.load(Mm_p[n], v);
            kernal_simd_aXpY(vs, Vm, Vd);
        }
        Vd.store(Vd_p, v);
    }
}

/**
 * @brief \todo TO BE FIXED
 * 
 * @tparam M 
 * @tparam N 
 * @tparam Tp 
 * @param A 
 * @param X 
 * @param Y 
 * @param gridSize 
 */

template <unsigned M, unsigned N, typename Tp>
void TensorGrid_complex_gemv_v2(const Tp *A, const Tp *X, Tp *Y, const size_t gridSize)
{
    typedef typename SimdTraits<vComplex<Tp>>::ptr_type CTp;

    CTp Vd_p[M];
    CTp Mm_p[M][N];
    CTp Vs_p[N];
    for (unsigned m = 0; m < M; m++) {
        for (unsigned n = 0; n < N; n++) {
            Mm_p[m][n][0] = const_cast<Tp *>(&A[(m * N * 2 + n * 2 + 0) * gridSize]);
            Mm_p[m][n][1] = const_cast<Tp *>(&A[(m * N * 2 + n * 2 + 1) * gridSize]);
        }
    }
    for (unsigned n = 0; n < N; n++) {
        Vs_p[n][0] = const_cast<Tp *>(&X[(n * 2 + 0) * gridSize]);
        Vs_p[n][1] = const_cast<Tp *>(&X[(n * 2 + 1) * gridSize]);
    }

    for (unsigned m = 0; m < M; m++) {
        Vd_p[m][0] = const_cast<Tp *>(&Y[(m * 2 + 0) * gridSize]);
        Vd_p[m][1] = const_cast<Tp *>(&Y[(m * 2 + 1) * gridSize]);
    }

    iVector<vComplex<Tp>, N> Vm;
    iVector<vComplex<Tp>, N> Vs;
    vComplex<Tp>             vdes;
    for (size_t v = 0; v < gridSize; v += vComplex<Tp>::NumElem) {
        Vs.load(Vs_p, v);
        for (unsigned m = 0; m < M; m++) {
            vdes.setzero();
            Vm.load(Mm_p[m], v);
            kernal_simd_XdotY(vdes, Vm, Vs);
            vdes.store(Vd_p[m], v);
        }
    }
}

#undef Version