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

#if defined(__AVX512F__)

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

    for (size_t v = 0; v < gridSize; v += 8) {
        for (size_t m = 0; m < M; m++) {
            __m512d vr_re = _mm512_setzero_pd();
            __m512d vr_im = _mm512_setzero_pd();
            for (size_t n = 0; n < N; n++) {
                __m512d mrc_re = _mm512_load_pd(&pm[m][n][0][v]);
                __m512d mrc_im = _mm512_load_pd(&pm[m][n][1][v]);
                __m512d vc_re  = _mm512_load_pd(&ps[n][0][v]);
                __m512d vc_im  = _mm512_load_pd(&ps[n][1][v]);

                vr_re = _mm512_add_pd(vr_re, _mm512_sub_pd(_mm512_mul_pd(mrc_re, vc_re),
                                                           _mm512_mul_pd(mrc_im, vc_im)));
                vr_im = _mm512_add_pd(vr_im, _mm512_add_pd(_mm512_mul_pd(mrc_re, vc_im),
                                                           _mm512_mul_pd(mrc_im, vc_re)));
            }
            _mm512_store_pd(&pd[m][0][v], vr_re);
            _mm512_store_pd(&pd[m][1][v], vr_im);
        }
    }

    // vComplexD vr;
    // for (size_t v = 0; v < gridSize; v += 8) {
    //     for (size_t m = 0; m < M; m++) {
    //         SimdSetzero(vr);
    //         for (size_t n = 0; n < N; n++) {
    //             vComplexD mcr = SimdLoad(&pm[m][n][0][v], &pm[m][n][1][v]);
    //             vComplexD vc  = SimdLoad(&ps[n][0][v], &ps[n][1][v]);
    //             // vr            = SimdAdd(vr, SimdMul(mcr, vc));
    //             vr            = SimdFmadd(mcr, vc, vr);
    //         }
    //         SimdStore(&pd[m][0][v], &pd[m][1][v], vr);
    //     }
    // }
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




/**
 * @brief 
 * 
 * @param dest 
 * @param mat 
 * @param src 
 * @param gridSize 
 */
void TensorGrid_CMatrixVector_avx512(double *dest, double *mat, double *src, size_t gridSize)
{
    double *pd[MAX_ROW][2];
    double *ps[NCOL][2];
    double *pm[MAX_ROW][NCOL][2];

    for (size_t col = 0; col < NCOL; col++) {
        ps[col][0] = src + col * 2 * gridSize;
        ps[col][1] = src + (col * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < MAX_ROW; row++) {
        pd[row][0] = dest + row * 2 * gridSize;
        pd[row][1] = dest + (row * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < NCOL; col++) {
            pm[row][col][0] = mat + (row * 2 * NCOL + 2 * col) * gridSize;
            pm[row][col][1] = mat + (row * 2 * NCOL + 2 * col + 1) * gridSize;
        }
    }

    for (size_t v = 0; v < gridSize; v += 8) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            __m512d vr_re = _mm512_setzero_pd();
            __m512d vr_im = _mm512_setzero_pd();
            for (size_t col = 0; col < NCOL; col++) {
                __m512d mrc_re = _mm512_load_pd(&pm[row][col][0][v]);
                __m512d mrc_im = _mm512_load_pd(&pm[row][col][1][v]);
                __m512d vc_re  = _mm512_load_pd(&ps[col][0][v]);
                __m512d vc_im  = _mm512_load_pd(&ps[col][1][v]);

                vr_re = _mm512_add_pd(vr_re, _mm512_sub_pd(_mm512_mul_pd(mrc_re, vc_re),
                                                           _mm512_mul_pd(mrc_im, vc_im)));
                vr_im = _mm512_add_pd(vr_im, _mm512_add_pd(_mm512_mul_pd(mrc_re, vc_im),
                                                           _mm512_mul_pd(mrc_im, vc_re)));
            }
            _mm512_store_pd(&pd[row][0][v], vr_re);
            _mm512_store_pd(&pd[row][1][v], vr_im);
        }
    }
}



/**
 * @brief 
 * 
 * @param dest 
 * @param mat 
 * @param src 
 * @param gridSize 
 */
void TensorGrid_CMatrixVector_avx512(float *dest, float *mat, float *src, size_t gridSize)
{
    float *pd[MAX_ROW][2];
    float *ps[NCOL][2];
    float *pm[MAX_ROW][NCOL][2];

    for (size_t col = 0; col < NCOL; col++) {
        ps[col][0] = src + col * 2 * gridSize;
        ps[col][1] = src + (col * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < MAX_ROW; row++) {
        pd[row][0] = dest + row * 2 * gridSize;
        pd[row][1] = dest + (row * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < NCOL; col++) {
            pm[row][col][0] = mat + (row * 2 * NCOL + 2 * col) * gridSize;
            pm[row][col][1] = mat + (row * 2 * NCOL + 2 * col + 1) * gridSize;
        }
    }

    for (size_t v = 0; v < gridSize; v += 16) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            __m512 vr_re = _mm512_setzero_ps();
            __m512 vr_im = _mm512_setzero_ps();
            for (size_t col = 0; col < NCOL; col++) {
                __m512 mrc_re = _mm512_load_ps(&pm[row][col][0][v]);
                __m512 mrc_im = _mm512_load_ps(&pm[row][col][1][v]);
                __m512 vc_re  = _mm512_load_ps(&ps[col][0][v]);
                __m512 vc_im  = _mm512_load_ps(&ps[col][1][v]);

                vr_re = _mm512_add_ps(vr_re, _mm512_sub_ps(_mm512_mul_ps(mrc_re, vc_re),
                                                           _mm512_mul_ps(mrc_im, vc_im)));
                vr_im = _mm512_add_ps(vr_im, _mm512_add_ps(_mm512_mul_ps(mrc_re, vc_im),
                                                           _mm512_mul_ps(mrc_im, vc_re)));
            }
            _mm512_store_ps(&pd[row][0][v], vr_re);
            _mm512_store_ps(&pd[row][1][v], vr_im);
        }
    }
}

#endif