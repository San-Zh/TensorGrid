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
#include <iostream>
// #include "TensorGrid_Blas_avx.h"
// #include "TensorGrid_Blas_avx512.h"
#include "SimdTypes.h"
#include <complex>

// clang-format off
template <typename T> struct PrecTraits;
template <> struct PrecTraits<float > { typedef float  value_type; };
template <> struct PrecTraits<double> { typedef double value_type; };
// clang-format on


template <typename Tp, int M, int N>
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
    for (size_t v = 0; v < gridSize; v+=16) { //= sizeof(vReal<Tp>)
        for (size_t m = 0; m < M; m++) { SimdSetzero(Vd.vec[m]); }

        for (size_t n = 0; n < N; n++) {
            vs = SimdLoad(&Vsp[n][0][v], &Vsp[n][1][v]); ///
            for (size_t m = 0; m < M; m++) {
                Mn.vec[m] = SimdLoad(&Mp[m][n][0][v], &Mp[m][n][1][v]); ///
                // Vd.vec[m] = SimdFmadd(Mn.vec[m], vs, Vd.vec[m]);
                Vd.vec[m] = SimdAdd(SimdMul(Mn.vec[m], vs), Vd.vec[m]);
            }
        }

        for (int m = 0; m < M; m++) { SimdStore(&Vdp[m][0][v], &Vdp[m][1][v], Vd.vec[m]); }
    }

    // std::complex<Tp> Mn[M];
    // std::complex<Tp> vs;
    // for (size_t v = 0; v < gridSize; v++) {
    //     // for (size_t m = 0; m < M; m++) {
    //     //     Vd[m].real(Vdp[m][0][v]);
    //     //     Vd[m].imag(Vdp[m][1][v]);
    //     // }
    //     std::complex<Tp> Vd[M] = {0};

    //     for (size_t n = 0; n < N; n++) {
    //         vs.real(Vsp[n][0][v]);
    //         vs.imag(Vsp[n][1][v]);
    //         for (size_t m = 0; m < M; m++) {
    //             Mn[m] = std::complex<Tp>(Mp[m][n][0][v], Mp[m][n][1][v]); ///
    //             Vd[m] = Mn[m] * vs + Vd[m];
    //         }
    //     }

    //     for (int m = 0; m < M; m++) { Vdp[m][0][v] = Vd[m].real(), Vdp[m][1][v] = Vd[m].imag(); }
    // }
}
