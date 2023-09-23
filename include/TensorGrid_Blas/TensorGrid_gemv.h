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


/**
 * @brief v0.0.0: TensorGrid_zgemv_batch
 * 
 * @tparam Tp 
 * @tparam M 
 * @tparam N 
 * @param dest 
 * @param mat 
 * @param src 
 * @param gridSize 
 */
template <typename Tp, unsigned M, unsigned N>
void TensorGrid_zgemv_batch(Tp *dest, const Tp *mat, const Tp *src, size_t gridSize)
{
    Tp *Mp[M][N][2];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            Mp[i][j][0] = const_cast<Tp *>(&mat[(i * N * 2 + j * 2 + 0) * gridSize]);
            Mp[i][j][1] = const_cast<Tp *>(&mat[(i * N * 2 + j * 2 + 1) * gridSize]);
        }
    }
    Tp *Vdp[M][2];
    for (int i = 0; i < M; i++) {
        Vdp[i][0] = &dest[(i * 2 + 0) * gridSize];
        Vdp[i][1] = &dest[(i * 2 + 1) * gridSize];
    }
    Tp *Vsp[N][2];
    for (int j = 0; j < N; j++) {
        Vsp[j][0] = const_cast<Tp *>(&src[(j * 2 + 0) * gridSize]);
        Vsp[j][1] = const_cast<Tp *>(&src[(j * 2 + 1) * gridSize]);
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
            SimdLoad(vs, &Vsp[n][0][v], &Vsp[n][1][v]); ///
            for (size_t m = 0; m < M; m++) {
                Mn[m] = SimdLoad(&Mp[m][n][0][v], &Mp[m][n][1][v]); ///
                Vd[m] = SimdFmadd(Mn[m], vs, Vd[m]);
                // SimdLoad(Mn(m), &Mp[m][n][0][v], &Mp[m][n][1][v]); ///
                // SimdFmadd(Vd(m), Mn(m), vs, Vd(m));
            }
        }

        for (int m = 0; m < M; m++) { SimdStore(&Vdp[m][0][v], &Vdp[m][1][v], Vd[m]); }
    }
}




/**
 * @brief v0.0.1 : outer product method
 * 
 * @tparam Tp 
 * @tparam M 
 * @tparam N 
 * @param dest 
 * @param mat 
 * @param src 
 * @param gridSize 
 */
template <typename Tp, unsigned M, unsigned N>
void TensorGrid_zgemv_batch_v1(Tp *dest, const Tp *mat, const Tp *src, size_t gridSize)
{
    typedef typename SimdTraits<vComplex<Tp>>::ptr_type CTp;

    CTp Mm_p[N][M];
    CTp Vd_p[M];
    CTp Vs_p[N];

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            Mm_p[j][i][0] = const_cast<Tp *>(&mat[(i * N * 2 + j * 2 + 0) * gridSize]);
            Mm_p[j][i][1] = const_cast<Tp *>(&mat[(i * N * 2 + j * 2 + 1) * gridSize]);
        }
    }
    for (int i = 0; i < M; i++) {
        Vd_p[i][0] = &dest[(i * 2 + 0) * gridSize];
        Vd_p[i][1] = &dest[(i * 2 + 1) * gridSize];
    }
    for (int j = 0; j < N; j++) {
        Vs_p[j][0] = const_cast<Tp *>(&src[(j * 2 + 0) * gridSize]);
        Vs_p[j][1] = const_cast<Tp *>(&src[(j * 2 + 1) * gridSize]);
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
 * @tparam Tp 
 * @tparam M 
 * @tparam N 
 * @param dest 
 * @param mat 
 * @param src 
 * @param gridSize 
 */

template <typename Tp, unsigned M, unsigned N>
void TensorGrid_zgemv_batch_v2(Tp *dest, const Tp *mat, const Tp *src, const size_t gridSize)
{
    typedef typename SimdTraits<vComplex<Tp>>::ptr_type CTp;

    CTp Vd_p[M];
    CTp Mm_p[M][N];
    CTp Vs_p[N];
    for (unsigned m = 0; m < M; m++) {
        for (unsigned n = 0; n < N; n++) {
            Mm_p[m][n][0] = const_cast<Tp *>(&mat[(m * N * 2 + n * 2 + 0) * gridSize]);
            Mm_p[m][n][1] = const_cast<Tp *>(&mat[(m * N * 2 + n * 2 + 1) * gridSize]);
        }
    }
    for (unsigned n = 0; n < N; n++) {
        Vs_p[n][0] = const_cast<Tp *>(&src[(n * 2 + 0) * gridSize]);
        Vs_p[n][1] = const_cast<Tp *>(&src[(n * 2 + 1) * gridSize]);
    }

    for (unsigned m = 0; m < M; m++) {
        Vd_p[m][0] = const_cast<Tp *>(&dest[(m * 2 + 0) * gridSize]);
        Vd_p[m][1] = const_cast<Tp *>(&dest[(m * 2 + 1) * gridSize]);
    }

    iVector<vComplex<Tp>, N> Vm;
    iVector<vComplex<Tp>, N> Vs;
    vComplex<Tp>             vdes;
    for (size_t v = 0; v < gridSize; v += vComplex<Tp>::NumElem) {
        Vs.load(Vs_p, v);
        for (unsigned m = 0; m < M; m++) {
            SimdSetzero(vdes);
            Vm.load(Mm_p[m], v);
            kernal_simd_XdotY(vdes, Vm, Vs);
            SimdStore(Vd_p[m], v, vdes);
        }
    }
}

#undef Version