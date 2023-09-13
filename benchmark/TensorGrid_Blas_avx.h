/**
 * @file TensorGrid_Blas_avx.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-10
 * 
 * @copyright Copyright (c) 2023
 * 
 */


#pragma once

#include <immintrin.h>

template <unsigned M, unsigned N>
void TensorGrid_CMatrixVector_Batch_avx256(double *dest, double *mat, double *src, size_t gridSize)
{
    double *pd[M][2];
    double *ps[N][2];
    double *pm[M][N][2];
    for (size_t col = 0; col < N; col++) {
        ps[col][0] = src + col * 2 * gridSize;
        ps[col][1] = src + (col * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < M; row++) {
        pd[row][0] = dest + row * 2 * gridSize;
        pd[row][1] = dest + (row * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < M; row++) {
        for (size_t col = 0; col < N; col++) {
            pm[row][col][0] = mat + (row * 2 * N + 2 * col) * gridSize;
            pm[row][col][1] = mat + (row * 2 * N + 2 * col + 1) * gridSize;
        }
    }

    for (size_t v = 0; v < gridSize; v += 4) {
        for (size_t row = 0; row < M; row++) {
            __m256d vr_re = _mm256_setzero_pd();
            __m256d vr_im = _mm256_setzero_pd();
            for (size_t col = 0; col < N; col++) {
                __m256d mrc_re = _mm256_load_pd((pm[row][col][0] + v));
                __m256d mrc_im = _mm256_load_pd(pm[row][col][1] + v);
                __m256d vc_re  = _mm256_load_pd(ps[col][0] + v);
                __m256d vc_im  = _mm256_load_pd(ps[col][1] + v);
                vr_re          = _mm256_add_pd(vr_re, _mm256_sub_pd(_mm256_mul_pd(mrc_re, vc_re),
                                                                    _mm256_mul_pd(mrc_im, vc_im)));
                vr_im          = _mm256_add_pd(vr_im, _mm256_add_pd(_mm256_mul_pd(mrc_re, vc_im),
                                                                    _mm256_mul_pd(mrc_im, vc_re)));
            }
            _mm256_store_pd(pd[row][0] + v, vr_re);
            _mm256_store_pd(pd[row][1] + v, vr_im);
        }
    }
}

template <unsigned M, unsigned N>
void TensorGrid_CMatrixVector_Batch_avx256(float *dest, float *mat, float *src, size_t gridSize)
{
    float *pd[M][2];
    float *ps[N][2];
    float *pm[M][N][2];
    for (size_t col = 0; col < N; col++) {
        ps[col][0] = src + col * 2 * gridSize;
        ps[col][1] = src + (col * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < M; row++) {
        pd[row][0] = dest + row * 2 * gridSize;
        pd[row][1] = dest + (row * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < M; row++) {
        for (size_t col = 0; col < N; col++) {
            pm[row][col][0] = mat + (row * 2 * N + 2 * col) * gridSize;
            pm[row][col][1] = mat + (row * 2 * N + 2 * col + 1) * gridSize;
        }
    }

    for (size_t v = 0; v < gridSize; v += 8) {
        for (size_t row = 0; row < M; row++) {
            __m256 vr_re = _mm256_setzero_ps();
            __m256 vr_im = _mm256_setzero_ps();
            // __m256 tmpmul;
            for (size_t col = 0; col < N; col++) {
                __m256 mrc_re = _mm256_load_ps(&(pm[row][col][0][v]));
                __m256 mrc_im = _mm256_load_ps(&(pm[row][col][1][v]));
                __m256 vc_re  = _mm256_load_ps(&(ps[col][0][v]));
                __m256 vc_im  = _mm256_load_ps(&(ps[col][1][v]));
                vr_re         = _mm256_add_ps(vr_re, _mm256_sub_ps(_mm256_mul_ps(mrc_re, vc_re),
                                                                   _mm256_mul_ps(mrc_im, vc_im)));
                vr_im         = _mm256_add_ps(vr_im, _mm256_add_ps(_mm256_mul_ps(mrc_re, vc_im),
                                                                   _mm256_mul_ps(mrc_im, vc_re)));
            }
            _mm256_store_ps((pd[row][0] + v), vr_re);
            _mm256_store_ps((pd[row][1] + v), vr_im);
        }
    }
}

template <unsigned M, unsigned N>
void TensorGrid_CMatrixVector_Batch_avx256_expand(double *dest, double *mat, double *src,
                                                  size_t gridSize)
{
    double *pd[M][2];
    double *ps[N][2];
    double *pm[M][N][2];
    for (size_t col = 0; col < N; col++) {
        ps[col][0] = src + col * 2 * gridSize;
        ps[col][1] = src + (col * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < M; row++) {
        pd[row][0] = dest + row * 2 * gridSize;
        pd[row][1] = dest + (row * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < M; row++) {
        for (size_t col = 0; col < N; col++) {
            pm[row][col][0] = mat + (row * 2 * N + 2 * col) * gridSize;
            pm[row][col][1] = mat + (row * 2 * N + 2 * col + 1) * gridSize;
        }
    }

    for (size_t v = 0; v < gridSize; v += 4) {
        __m256d vc0_re = _mm256_load_pd(&(ps[0][0][v]));
        __m256d vc0_im = _mm256_load_pd(&(ps[0][1][v]));
        __m256d vc1_re = _mm256_load_pd(&(ps[1][0][v]));
        __m256d vc1_im = _mm256_load_pd(&(ps[1][1][v]));
        __m256d vc2_re = _mm256_load_pd(&(ps[2][0][v]));
        __m256d vc2_im = _mm256_load_pd(&(ps[2][1][v]));

        for (size_t row = 0; row < M; row++) {
            // for (size_t col = 0; col < N; col++) {
            __m256d vr_re  = _mm256_setzero_pd();
            __m256d vr_im  = _mm256_setzero_pd();
            __m256d mr0_re = _mm256_load_pd(&(pm[row][0][0][v]));
            __m256d mr0_im = _mm256_load_pd(&(pm[row][0][1][v]));
            __m256d mr1_re = _mm256_load_pd(&(pm[row][1][0][v]));
            __m256d mr1_im = _mm256_load_pd(&(pm[row][1][1][v]));
            __m256d mr2_re = _mm256_load_pd(&(pm[row][2][0][v]));
            __m256d mr2_im = _mm256_load_pd(&(pm[row][2][1][v]));
            vr_re          = _mm256_add_pd(
                         vr_re, _mm256_sub_pd(_mm256_mul_pd(mr0_re, vc0_re), _mm256_mul_pd(mr0_im, vc0_im)));
            vr_im = _mm256_add_pd(
                vr_im, _mm256_add_pd(_mm256_mul_pd(mr0_re, vc0_im), _mm256_mul_pd(mr0_im, vc0_re)));
            vr_re = _mm256_add_pd(
                vr_re, _mm256_sub_pd(_mm256_mul_pd(mr1_re, vc1_re), _mm256_mul_pd(mr1_im, vc1_im)));
            vr_im = _mm256_add_pd(
                vr_im, _mm256_add_pd(_mm256_mul_pd(mr1_re, vc1_im), _mm256_mul_pd(mr1_im, vc1_re)));
            vr_re = _mm256_add_pd(
                vr_re, _mm256_sub_pd(_mm256_mul_pd(mr2_re, vc2_re), _mm256_mul_pd(mr2_im, vc2_im)));
            vr_im = _mm256_add_pd(
                vr_im, _mm256_add_pd(_mm256_mul_pd(mr2_re, vc2_im), _mm256_mul_pd(mr2_im, vc2_re)));
            _mm256_store_pd((pd[row][0] + v), vr_re);
            _mm256_store_pd((pd[row][1] + v), vr_im);
            // }
        }
    }
}

template <unsigned M, unsigned N>
void TensorGrid_CMatrixVector_Batch_avx256_expand(float *dest, float *mat, float *src,
                                                  size_t gridSize)
{
    float *pd[M][2];
    float *ps[N][2];
    float *pm[M][N][2];
    for (size_t col = 0; col < N; col++) {
        ps[col][0] = src + col * 2 * gridSize;
        ps[col][1] = src + (col * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < M; row++) {
        pd[row][0] = dest + row * 2 * gridSize;
        pd[row][1] = dest + (row * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < M; row++) {
        for (size_t col = 0; col < N; col++) {
            pm[row][col][0] = mat + (row * 2 * N + 2 * col) * gridSize;
            pm[row][col][1] = mat + (row * 2 * N + 2 * col + 1) * gridSize;
        }
    }

    __m256 vc0_re;
    __m256 vc0_im;
    __m256 vc1_re;
    __m256 vc1_im;
    __m256 vc2_re;
    __m256 vc2_im;

    __m256 mr0_re;
    __m256 mr0_im;
    __m256 mr1_re;
    __m256 mr1_im;
    __m256 mr2_re;
    __m256 mr2_im;

    __m256 vr_re;
    __m256 vr_im;

    for (size_t v = 0; v < gridSize; v += 8) {
        vc0_re = _mm256_load_ps(&(ps[0][0][v]));
        vc0_im = _mm256_load_ps(&(ps[0][1][v]));
        vc1_re = _mm256_load_ps(&(ps[1][0][v]));
        vc1_im = _mm256_load_ps(&(ps[1][1][v]));
        vc2_re = _mm256_load_ps(&(ps[2][0][v]));
        vc2_im = _mm256_load_ps(&(ps[2][1][v]));

        for (size_t row = 0; row < M; row++) {
            // for (size_t col = 0; col < N; col++) {
            vr_re  = _mm256_setzero_ps();
            vr_im  = _mm256_setzero_ps();
            mr0_re = _mm256_load_ps(&(pm[row][0][0][v]));
            mr0_im = _mm256_load_ps(&(pm[row][0][1][v]));
            vr_re  = _mm256_add_ps(
                 vr_re, _mm256_sub_ps(_mm256_mul_ps(mr0_re, vc0_re), _mm256_mul_ps(mr0_im, vc0_im)));
            vr_im = _mm256_add_ps(
                vr_im, _mm256_add_ps(_mm256_mul_ps(mr0_re, vc0_im), _mm256_mul_ps(mr0_im, vc0_re)));
            mr1_re = _mm256_load_ps(&(pm[row][1][0][v]));
            mr1_im = _mm256_load_ps(&(pm[row][1][1][v]));
            vr_re  = _mm256_add_ps(
                 vr_re, _mm256_sub_ps(_mm256_mul_ps(mr1_re, vc1_re), _mm256_mul_ps(mr1_im, vc1_im)));
            vr_im = _mm256_add_ps(
                vr_im, _mm256_add_ps(_mm256_mul_ps(mr1_re, vc1_im), _mm256_mul_ps(mr1_im, vc1_re)));
            mr2_re = _mm256_load_ps(&(pm[row][2][0][v]));
            mr2_im = _mm256_load_ps(&(pm[row][2][1][v]));
            vr_re  = _mm256_add_ps(
                 vr_re, _mm256_sub_ps(_mm256_mul_ps(mr2_re, vc2_re), _mm256_mul_ps(mr2_im, vc2_im)));
            vr_im = _mm256_add_ps(
                vr_im, _mm256_add_ps(_mm256_mul_ps(mr2_re, vc2_im), _mm256_mul_ps(mr2_im, vc2_re)));
            _mm256_store_ps((pd[row][0] + v), vr_re);
            _mm256_store_ps((pd[row][1] + v), vr_im);

            // }
        }
    }
}