
#pragma once

#include <cstdlib>
#include <complex>
#include <immintrin.h>
#include "setup.h"

#if defined __AVX__
void TensorGrid_CMatrixVector_avx256(double *dest, double *mat, double *src, size_t gridSize)
{
    double *pd[MAX_ROW][2];
    double *ps[MAX_COL][2];
    double *pm[MAX_ROW][MAX_COL][2];
    for (size_t col = 0; col < MAX_COL; col++) {
        ps[col][0] = src + col * 2 * gridSize;
        ps[col][1] = src + (col * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < MAX_ROW; row++) {
        pd[row][0] = dest + row * 2 * gridSize;
        pd[row][1] = dest + (row * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            pm[row][col][0] = mat + (row * 2 * MAX_COL + 2 * col) * gridSize;
            pm[row][col][1] = mat + (row * 2 * MAX_COL + 2 * col + 1) * gridSize;
        }
    }

    for (size_t v = 0; v < gridSize; v += 4) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            __m256d vr_re = _mm256_setzero_pd();
            __m256d vr_im = _mm256_setzero_pd();
            for (size_t col = 0; col < MAX_COL; col++) {
                __m256d mrc_re = _mm256_load_pd((pm[row][col][0] + v));
                __m256d mrc_im = _mm256_load_pd(pm[row][col][1] + v);
                __m256d vc_re = _mm256_load_pd(ps[col][0] + v);
                __m256d vc_im = _mm256_load_pd(ps[col][1] + v);
                vr_re = _mm256_add_pd(vr_re, _mm256_sub_pd(_mm256_mul_pd(mrc_re, vc_re), _mm256_mul_pd(mrc_im, vc_im)));
                vr_im = _mm256_add_pd(vr_im, _mm256_add_pd(_mm256_mul_pd(mrc_re, vc_im), _mm256_mul_pd(mrc_im, vc_re)));
            }
            _mm256_store_pd(pd[row][0] + v, vr_re);
            _mm256_store_pd(pd[row][1] + v, vr_im);
        }
    }
}

void TensorGrid_CMatrixVector_avx256(float *dest, float *mat, float *src, size_t gridSize)
{
    float *pd[MAX_ROW][2];
    float *ps[MAX_COL][2];
    float *pm[MAX_ROW][MAX_COL][2];
    for (size_t col = 0; col < MAX_COL; col++) {
        ps[col][0] = src + col * 2 * gridSize;
        ps[col][1] = src + (col * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < MAX_ROW; row++) {
        pd[row][0] = dest + row * 2 * gridSize;
        pd[row][1] = dest + (row * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            pm[row][col][0] = mat + (row * 2 * MAX_COL + 2 * col) * gridSize;
            pm[row][col][1] = mat + (row * 2 * MAX_COL + 2 * col + 1) * gridSize;
        }
    }

    for (size_t v = 0; v < gridSize; v += 8) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            __m256 vr_re = _mm256_setzero_ps();
            __m256 vr_im = _mm256_setzero_ps();
            // __m256 tmpmul;
            for (size_t col = 0; col < MAX_COL; col++) {
                __m256 mrc_re = _mm256_load_ps(&(pm[row][col][0][v]));
                __m256 mrc_im = _mm256_load_ps(&(pm[row][col][1][v]));
                __m256 vc_re = _mm256_load_ps(&(ps[col][0][v]));
                __m256 vc_im = _mm256_load_ps(&(ps[col][1][v]));
                vr_re = _mm256_add_ps(vr_re, _mm256_sub_ps(_mm256_mul_ps(mrc_re, vc_re), _mm256_mul_ps(mrc_im, vc_im)));
                vr_im = _mm256_add_ps(vr_im, _mm256_add_ps(_mm256_mul_ps(mrc_re, vc_im), _mm256_mul_ps(mrc_im, vc_re)));
            }
            _mm256_store_ps((pd[row][0] + v), vr_re);
            _mm256_store_ps((pd[row][1] + v), vr_im);
        }
    }
}

#endif

#ifdef __AVX512F__

void TensorGrid_CMatrixVector_avx512(double *dest, double *mat, double *src, size_t gridSize)
{
    double *pd[MAX_ROW][2];
    double *ps[MAX_COL][2];
    double *pm[MAX_ROW][MAX_COL][2];

    for (size_t col = 0; col < MAX_COL; col++) {
        ps[col][0] = src + col * 2 * gridSize;
        ps[col][1] = src + (col * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < MAX_ROW; row++) {
        pd[row][0] = dest + row * 2 * gridSize;
        pd[row][1] = dest + (row * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            pm[row][col][0] = mat + (row * 2 * MAX_COL + 2 * col) * gridSize;
            pm[row][col][1] = mat + (row * 2 * MAX_COL + 2 * col + 1) * gridSize;
        }
    }

    for (size_t v = 0; v < gridSize; v += 8) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            __m512d vr_re = _mm512_setzero_pd();
            __m512d vr_im = _mm512_setzero_pd();
            for (size_t col = 0; col < MAX_COL; col++) {
                __m512d mrc_re = _mm512_load_pd(&pm[row][col][0][v]);
                __m512d mrc_im = _mm512_load_pd(&pm[row][col][1][v]);
                __m512d vc_re = _mm512_load_pd(&ps[col][0][v]);
                __m512d vc_im = _mm512_load_pd(&ps[col][1][v]);
                vr_re = _mm512_add_pd(vr_re, _mm512_sub_pd(_mm512_mul_pd(mrc_re, vc_re), _mm512_mul_pd(mrc_im, vc_im)));
                vr_im = _mm512_add_pd(vr_im, _mm512_add_pd(_mm512_mul_pd(mrc_re, vc_im), _mm512_mul_pd(mrc_im, vc_re)));
            }
            _mm512_store_pd(&pd[row][0][v], vr_re);
            _mm512_store_pd(&pd[row][1][v], vr_im);
        }
    }
}

void TensorGrid_CMatrixVector_avx512(float *dest, float *mat, float *src, size_t gridSize)
{
    float *pd[MAX_ROW][2];
    float *ps[MAX_COL][2];
    float *pm[MAX_ROW][MAX_COL][2];

    for (size_t col = 0; col < MAX_COL; col++) {
        ps[col][0] = src + col * 2 * gridSize;
        ps[col][1] = src + (col * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < MAX_ROW; row++) {
        pd[row][0] = dest + row * 2 * gridSize;
        pd[row][1] = dest + (row * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            pm[row][col][0] = mat + (row * 2 * MAX_COL + 2 * col) * gridSize;
            pm[row][col][1] = mat + (row * 2 * MAX_COL + 2 * col + 1) * gridSize;
        }
    }

    for (size_t v = 0; v < gridSize; v += 16) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            __m512 vr_re = _mm512_setzero_ps();
            __m512 vr_im = _mm512_setzero_ps();
            for (size_t col = 0; col < MAX_COL; col++) {
                __m512 mrc_re = _mm512_load_ps(&pm[row][col][0][v]);
                __m512 mrc_im = _mm512_load_ps(&pm[row][col][1][v]);
                __m512 vc_re = _mm512_load_ps(&ps[col][0][v]);
                __m512 vc_im = _mm512_load_ps(&ps[col][1][v]);
                vr_re = _mm512_add_ps(vr_re, _mm512_sub_ps(_mm512_mul_ps(mrc_re, vc_re), _mm512_mul_ps(mrc_im, vc_im)));
                vr_im = _mm512_add_ps(vr_im, _mm512_add_ps(_mm512_mul_ps(mrc_re, vc_im), _mm512_mul_ps(mrc_im, vc_re)));
            }
            _mm512_store_ps(&pd[row][0][v], vr_re);
            _mm512_store_ps(&pd[row][1][v], vr_im);
        }
    }
}

#endif