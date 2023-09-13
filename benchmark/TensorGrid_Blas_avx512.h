/**
 * @file TensorGrid_Blas_avx512.h
 * @author your name (you@domain.com)
 * @brief TensorGrid_CMatrix_Batch_avx512(_fma/expand)*<M,N>  use avx512f
 * @version 0.1
 * @date 2023-09-12
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once

#include <immintrin.h>

template <unsigned M, unsigned N>
void TensorGrid_CMatrixVector_Batch_avx512(double *dest, double *mat, double *src, size_t gridSize)
{
    double *pd[M][2];
    double *ps[N][2];
    double *pm[M][N][2];

    for (size_t n = 0; n < N; n++) {
        ps[n][0] = src + (n * 2) * gridSize;
        ps[n][1] = src + (n * 2 + 1) * gridSize;
    }
    for (size_t m = 0; m < M; m++) {
        pd[m][0] = dest + (m * 2) * gridSize;
        pd[m][1] = dest + (m * 2 + 1) * gridSize;
    }
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            pm[m][n][0] = mat + (m * 2 * N + 2 * n) * gridSize;
            pm[m][n][1] = mat + (m * 2 * N + 2 * n + 1) * gridSize;
        }
    }

    // for (size_t v = 0; v < gridSize; v += 8) {
    //     for (size_t m = 0; m < M; m++) {
    //         __m512d vr_re = _mm512_setzero_pd();
    //         __m512d vr_im = _mm512_setzero_pd();
    //         for (size_t n = 0; n < N; n++) {
    //             __m512d mrc_re = _mm512_load_pd(&pm[m][n][0][v]);
    //             __m512d mrc_im = _mm512_load_pd(&pm[m][n][1][v]);
    //             __m512d vc_re  = _mm512_load_pd(&ps[n][0][v]);
    //             __m512d vc_im  = _mm512_load_pd(&ps[n][1][v]);

    //             vr_re = _mm512_add_pd(vr_re, _mm512_sub_pd(_mm512_mul_pd(mrc_re, vc_re),
    //                                                        _mm512_mul_pd(mrc_im, vc_im)));
    //             vr_im = _mm512_add_pd(vr_im, _mm512_add_pd(_mm512_mul_pd(mrc_re, vc_im),
    //                                                        _mm512_mul_pd(mrc_im, vc_re)));
    //         }
    //         _mm512_store_pd(&pd[m][0][v], vr_re);
    //         _mm512_store_pd(&pd[m][1][v], vr_im);
    //     }
    // }

    vComplexD vr;
    for (size_t v = 0; v < gridSize; v += 8) {
        for (size_t m = 0; m < M; m++) {
            SimdSetzero(vr);
            for (size_t n = 0; n < N; n++) {
                vComplexD mcr = SimdLoad(&pm[m][n][0][v], &pm[m][n][1][v]);
                vComplexD vc  = SimdLoad(&ps[n][0][v], &ps[n][1][v]);
                // vr            = SimdAdd(vr, SimdMul(mcr, vc));
                vr            = SimdFmadd(mcr, vc, vr);
            }
            SimdStore(&pd[m][0][v], &pd[m][1][v], vr);
        }
    }
}


template <unsigned M, unsigned N>
void TensorGrid_CMatrixVector_Batch_avx512_fma(double *dest, double *mat, double *src,
                                               const size_t &gridSize)
{
    double *pd[M][2];
    double *ps[N][2];
    double *pm[M][N][2];

    for (size_t n = 0; n < N; n++) {
        ps[n][0] = src + (n * 2) * gridSize;
        ps[n][1] = src + (n * 2 + 1) * gridSize;
    }
    for (size_t m = 0; m < M; m++) {
        pd[m][0] = dest + (m * 2) * gridSize;
        pd[m][1] = dest + (m * 2 + 1) * gridSize;
    }
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            pm[m][n][0] = mat + (m * 2 * N + 2 * n) * gridSize;
            pm[m][n][1] = mat + (m * 2 * N + 2 * n + 1) * gridSize;
        }
    }

    for (size_t v = 0; v < gridSize; v += 8) {
        for (size_t m = 0; m < M; m++) {
            __m512d vr_re = _mm512_setzero_pd();
            __m512d vr_im = _mm512_setzero_pd();
            for (size_t n = 0; n < N; n++) {
                __m512d mrc_re = _mm512_load_pd(&pm[m][n][0][v]);
                __m512d mrc_im = _mm512_load_pd(&pm[m][n][1][v]);
                __m512d vc_re  = _mm512_load_pd(&ps[n][0][v]);
                __m512d vc_im  = _mm512_load_pd(&ps[n][1][v]);

                vr_re = _mm512_fmsub_pd(mrc_re, vc_re, _mm512_fmsub_pd(mrc_im, vc_im, vr_re));
                vr_im = _mm512_fmadd_pd(mrc_re, vc_im, _mm512_fmadd_pd(mrc_im, vc_re, vr_im));
            }
            _mm512_store_pd(&pd[m][0][v], vr_re);
            _mm512_store_pd(&pd[m][1][v], vr_im);
        }
    }
}

template <unsigned M, unsigned N>
void TensorGrid_CMatrixVector_Batch_avx512(float *dest, float *mat, float *src, size_t gridSize)
{
    float *pd[M][2];
    float *ps[N][2];
    float *pm[M][N][2];

    for (size_t n = 0; n < N; n++) {
        ps[n][0] = src + (n * 2) * gridSize;
        ps[n][1] = src + (n * 2 + 1) * gridSize;
    }
    for (size_t m = 0; m < M; m++) {
        pd[m][0] = dest + (m * 2) * gridSize;
        pd[m][1] = dest + (m * 2 + 1) * gridSize;
    }
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            pm[m][n][0] = mat + (m * 2 * N + 2 * n) * gridSize;
            pm[m][n][1] = mat + (m * 2 * N + 2 * n + 1) * gridSize;
        }
    }

    for (size_t v = 0; v < gridSize; v += 16) {
        for (size_t m = 0; m < M; m++) {
            __m512 vr_re = _mm512_setzero_ps();
            __m512 vr_im = _mm512_setzero_ps();
            for (size_t n = 0; n < N; n++) {
                __m512 mrc_re = _mm512_load_ps(&pm[m][n][0][v]);
                __m512 mrc_im = _mm512_load_ps(&pm[m][n][1][v]);
                __m512 vc_re  = _mm512_load_ps(&ps[n][0][v]);
                __m512 vc_im  = _mm512_load_ps(&ps[n][1][v]);

                vr_re = _mm512_add_ps(vr_re, _mm512_sub_ps(_mm512_mul_ps(mrc_re, vc_re),
                                                           _mm512_mul_ps(mrc_im, vc_im)));
                vr_im = _mm512_add_ps(vr_im, _mm512_add_ps(_mm512_mul_ps(mrc_re, vc_im),
                                                           _mm512_mul_ps(mrc_im, vc_re)));
            }
            _mm512_store_ps(&pd[m][0][v], vr_re);
            _mm512_store_ps(&pd[m][1][v], vr_im);
        }
    }
}


template <unsigned M, unsigned N>
void TensorGrid_CMatrixVector_Batch_avx512_fma(float *dest, float *mat, float *src, size_t gridSize)
{
    float *pd[M][2];
    float *ps[N][2];
    float *pm[M][N][2];

    for (size_t n = 0; n < N; n++) {
        ps[n][0] = src + (n * 2) * gridSize;
        ps[n][1] = src + (n * 2 + 1) * gridSize;
    }
    for (size_t m = 0; m < M; m++) {
        pd[m][0] = dest + (m * 2) * gridSize;
        pd[m][1] = dest + (m * 2 + 1) * gridSize;
    }
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            pm[m][n][0] = mat + (m * 2 * N + 2 * n) * gridSize;
            pm[m][n][1] = mat + (m * 2 * N + 2 * n + 1) * gridSize;
        }
    }

    for (size_t v = 0; v < gridSize; v += 16) {
        for (size_t m = 0; m < M; m++) {
            __m512 vr_re = _mm512_setzero_ps();
            __m512 vr_im = _mm512_setzero_ps();
            for (size_t n = 0; n < N; n++) {
                __m512 mrc_re = _mm512_load_ps(&pm[m][n][0][v]);
                __m512 mrc_im = _mm512_load_ps(&pm[m][n][1][v]);
                __m512 vc_re  = _mm512_load_ps(&ps[n][0][v]);
                __m512 vc_im  = _mm512_load_ps(&ps[n][1][v]);

                vr_re = _mm512_fmsub_ps(mrc_re, vc_re, _mm512_fmsub_ps(mrc_im, vc_im, vr_re));
                vr_im = _mm512_fmadd_ps(mrc_re, vc_im, _mm512_fmadd_ps(mrc_im, vc_re, vr_im));
            }
            _mm512_store_ps(&pd[m][0][v], vr_re);
            _mm512_store_ps(&pd[m][1][v], vr_im);
        }
    }
}


template <unsigned M, unsigned N>
void TensorGrid_CMatrixVector_Batch_avx512_expand(double *dest, double *mat, double *src,
                                                  size_t gridSize)
{
    double *pd[M][2];
    double *ps[N][2];
    double *pm[M][N][2];
    for (size_t n = 0; n < N; n++) {
        ps[n][0] = src + (n * 2) * gridSize;
        ps[n][1] = src + (n * 2 + 1) * gridSize;
    }
    for (size_t m = 0; m < M; m++) {
        pd[m][0] = dest + (m * 2) * gridSize;
        pd[m][1] = dest + (m * 2 + 1) * gridSize;
    }
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            pm[m][n][0] = mat + (m * 2 * N + 2 * n) * gridSize;
            pm[m][n][1] = mat + (m * 2 * N + 2 * n + 1) * gridSize;
        }
    }
    __m512d ret[M][2];
    for (size_t m = 0; m < M; m++) {
        ret[m][0] = _mm512_setzero_pd();
        ret[m][1] = _mm512_setzero_pd();
    }

    for (size_t v = 0; v < gridSize; v += 8) {
        for (size_t n = 0; n < N; n++) {
            __m512d vc_re = _mm512_load_pd(&ps[n][0][v]);
            __m512d vc_im = _mm512_load_pd(&ps[n][1][v]);
            for (size_t m = 0; m < M; m++) {
                __m512d mrc_re = _mm512_load_pd(&pm[m][n][0][v]);
                __m512d mrc_im = _mm512_load_pd(&pm[m][n][1][v]);
                ret[n][0]      = _mm512_add_pd(
                         _mm512_sub_pd(_mm512_mul_pd(mrc_re, vc_re), _mm512_mul_pd(mrc_im, vc_im)),
                         ret[n][0]);
                ret[n][1] = _mm512_add_pd(
                    _mm512_add_pd(_mm512_mul_pd(mrc_re, vc_im), _mm512_mul_pd(mrc_im, vc_re)),
                    ret[n][1]);
            }
        }
        for (size_t m = 0; m < M; m++) {
            _mm512_store_pd(&pd[m][0][v], ret[m][0]);
            _mm512_store_pd(&pd[m][1][v], ret[m][1]);
        }
    }
}

template <unsigned M, unsigned N>
void TensorGrid_CMatrixVector_Batch_avx512_expand(float *dest, float *mat, float *src,
                                                  size_t gridSize)
{
    float *pd[M][2];
    float *ps[N][2];
    float *pm[M][N][2];
    for (size_t n = 0; n < N; n++) {
        ps[n][0] = src + (n * 2) * gridSize;
        ps[n][1] = src + (n * 2 + 1) * gridSize;
    }
    for (size_t m = 0; m < M; m++) {
        pd[m][0] = dest + (m * 2) * gridSize;
        pd[m][1] = dest + (m * 2 + 1) * gridSize;
    }
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            pm[m][n][0] = mat + (m * 2 * N + 2 * n) * gridSize;
            pm[m][n][1] = mat + (m * 2 * N + 2 * n + 1) * gridSize;
        }
    }

    for (size_t v = 0; v < gridSize; v += 16) {
        __m512 vc0_re = _mm512_load_ps(&(ps[0][0][v]));
        __m512 vc0_im = _mm512_load_ps(&(ps[0][1][v]));
        __m512 vc1_re = _mm512_load_ps(&(ps[1][0][v]));
        __m512 vc1_im = _mm512_load_ps(&(ps[1][1][v]));
        __m512 vc2_re = _mm512_load_ps(&(ps[2][0][v]));
        __m512 vc2_im = _mm512_load_ps(&(ps[2][1][v]));

        for (size_t m = 0; m < M; m++) {
            // for (size_t n = 0; n < N; n++) {
            __m512 vr_re  = _mm512_setzero_ps();
            __m512 vr_im  = _mm512_setzero_ps();
            __m512 mr0_re = _mm512_load_ps(&(pm[m][0][0][v]));
            __m512 mr0_im = _mm512_load_ps(&(pm[m][0][1][v]));
            __m512 mr1_re = _mm512_load_ps(&(pm[m][1][0][v]));
            __m512 mr1_im = _mm512_load_ps(&(pm[m][1][1][v]));
            __m512 mr2_re = _mm512_load_ps(&(pm[m][2][0][v]));
            __m512 mr2_im = _mm512_load_ps(&(pm[m][2][1][v]));
            vr_re         = _mm512_add_ps(
                        vr_re, _mm512_sub_ps(_mm512_mul_ps(mr0_re, vc0_re), _mm512_mul_ps(mr0_im, vc0_im)));
            vr_im = _mm512_add_ps(
                vr_im, _mm512_add_ps(_mm512_mul_ps(mr0_re, vc0_im), _mm512_mul_ps(mr0_im, vc0_re)));
            vr_re = _mm512_add_ps(
                vr_re, _mm512_sub_ps(_mm512_mul_ps(mr1_re, vc1_re), _mm512_mul_ps(mr1_im, vc1_im)));
            vr_im = _mm512_add_ps(
                vr_im, _mm512_add_ps(_mm512_mul_ps(mr1_re, vc1_im), _mm512_mul_ps(mr1_im, vc1_re)));
            vr_re = _mm512_add_ps(
                vr_re, _mm512_sub_ps(_mm512_mul_ps(mr2_re, vc2_re), _mm512_mul_ps(mr2_im, vc2_im)));
            vr_im = _mm512_add_ps(
                vr_im, _mm512_add_ps(_mm512_mul_ps(mr2_re, vc2_im), _mm512_mul_ps(mr2_im, vc2_re)));
            _mm512_store_ps((pd[m][0] + v), vr_re);
            _mm512_store_ps((pd[m][1] + v), vr_im);
            // }
        }
    }
}