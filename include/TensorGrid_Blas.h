/**
 * @file TensorGrid_Blas.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-10
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include "Simd/SimdTupleTypes.h"

// clang-format off
template <typename T> struct PrecTraits;
template <> struct PrecTraits<float > { typedef float  value_type; };
template <> struct PrecTraits<double> { typedef double value_type; };
// clang-format on

#define Version 0

#if (Version == 0)
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
    iVector<vComplex<Tp>, M> Vd;
    iVector<vComplex<Tp>, M> Mn;
    vComplex<Tp>             vs;
    for (size_t v = 0; v < gridSize; v += vComplex<Tp>::NumElem) { //= sizeof(vReal<Tp>)
        for (size_t m = 0; m < M; m++) { SimdSetzero(Vd[m]); }
        for (size_t n = 0; n < N; n++) {
            SimdLoad(vs, &Vsp[n][0][v], &Vsp[n][1][v]); ///
            for (size_t m = 0; m < M; m++) {
                Mn[m] = SimdLoad(&Mp[m][n][0][v], &Mp[m][n][1][v]); ///
                Vd[m] = SimdFmadd(Mn[m], vs, Vd[m]);
                // SimdLoad(Mn[m], &Mp[m][n][0][v], &Mp[m][n][1][v]); ///
                // SimdFmadd(Vd[m], Mn[m], vs, Vd[m]);
            }
        }

        for (int m = 0; m < M; m++) { SimdStore(&Vdp[m][0][v], &Vdp[m][1][v], Vd[m]); }
    }
}

#endif

#if (Version)
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
    // Tp *Mp[N][M][2];
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         Mp[j][i][0] = const_cast<Tp *>(&mat[(i * N * 2 + j * 2 + 0) * gridSize]);
    //         Mp[j][i][1] = const_cast<Tp *>(&mat[(i * N * 2 + j * 2 + 1) * gridSize]);
    //     }
    // }
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
    iVector<vComplex<Tp>, M> Vd;
    iVector<vComplex<Tp>, M> Mn;
    vComplex<Tp>             vs;
    for (size_t v = 0; v < gridSize; v += vComplex<Tp>::NumElem) { //= sizeof(vReal<Tp>)
        for (size_t m = 0; m < M; m++) { SimdSetzero(Vd[m]); }
        for (size_t n = 0; n < N; n++) {
            for (size_t m = 0; m < M; m++) { SimdLoad(Mn[m], &Mp[m][n][0][v], &Mp[m][n][1][v]); }
            SimdLoad(vs, &Vsp[n][0][v], &Vsp[n][1][v]); ///
            iVectorLoad(Mn, Mp[n]);                     ///
            for (size_t m = 0; m < M; m++) {
                // Mn[m] = SimdLoad(&Mp[m][n][0][v], &Mp[m][n][1][v]); ///
                // SimdLoad(Mn.vec[m], &Mp[m][n][0][v], &Mp[m][n][1][v]); ///
                Vd[m] = SimdFmadd(Mn[m], vs, Vd[m]);
            }
        }

        for (int m = 0; m < M; m++) { SimdStore(&Vdp[m][0][v], &Vdp[m][1][v], Vd[m]); }
    }
}

#endif

#undef Version