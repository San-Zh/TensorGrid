/**
 * @file benchmark_TensorGridBlas_avx256.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-16
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include <immintrin.h>



#define Debug(e)                                                 \
    do {                                                         \
        printf("debug: " #e " : %s : %d\n", __FILE__, __LINE__); \
    } while (0)



/**
 * @brief v0.0.2: version 0.0.2, TensorGrid_CMatrixVector_avx256; float prec
 * \note add M, N template params; and out-producted method performed
 * @tparam M 
 * @tparam N 
 * @param dest 
 * @param mat 
 * @param src 
 * @param gridSize 
 */
template <unsigned M, unsigned N>
void TensorGrid_CMatrixVector_avx256_v1(double *dest, double *mat, double *src, size_t gridSize)
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

    __m256d vd_re[M], vd_im[M];
    for (size_t v = 0; v < gridSize; v += 4) {
        for (size_t m = 0; m < M; m++) {
            vd_re[m] = _mm256_setzero_pd();
            vd_im[m] = _mm256_setzero_pd();
        }
        for (size_t col = 0; col < N; col++) {
            __m256d vc_re = _mm256_load_pd(ps[col][0] + v);
            __m256d vc_im = _mm256_load_pd(ps[col][1] + v);
            for (size_t row = 0; row < M; row++) {
                __m256d mrc_re = _mm256_load_pd(pm[row][col][0] + v);
                __m256d mrc_im = _mm256_load_pd(pm[row][col][1] + v);

                vd_re[row] = _mm256_add_pd(vd_re[row], _mm256_sub_pd(_mm256_mul_pd(mrc_re, vc_re),
                                                                     _mm256_mul_pd(mrc_im, vc_im)));
                vd_im[row] = _mm256_add_pd(vd_im[row], _mm256_add_pd(_mm256_mul_pd(mrc_re, vc_im),
                                                                     _mm256_mul_pd(mrc_im, vc_re)));
            }
            for (size_t m = 0; m < M; m++) {
                _mm256_store_pd(&pd[m][0][v], vd_re[m]);
                _mm256_store_pd(&pd[m][1][v], vd_im[m]);
            }
        }
    }
}

/**
 * @brief v0.0.2: version 0.0.2, TensorGrid_CMatrixVector_avx256; float prec
 * 
 * @tparam M 
 * @tparam N 
 * @param dest 
 * @param mat 
 * @param src 
 * @param gridSize 
 */
template <unsigned M, unsigned N>
void TensorGrid_CMatrixVector_avx256_v1(float *dest, float *mat, float *src, size_t gridSize)
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

    __m256 vd_re[M], vd_im[M];
    for (size_t v = 0; v < gridSize; v += 8) {
        for (size_t m = 0; m < M; m++) {
            vd_re[m] = _mm256_setzero_ps();
            vd_im[m] = _mm256_setzero_ps();
        }
        for (size_t col = 0; col < N; col++) {
            __m256 vc_re = _mm256_load_ps(ps[col][0] + v);
            __m256 vc_im = _mm256_load_ps(ps[col][1] + v);
            for (size_t row = 0; row < M; row++) {
                __m256 mrc_re = _mm256_load_ps(pm[row][col][0] + v);
                __m256 mrc_im = _mm256_load_ps(pm[row][col][1] + v);

                vd_re[row] = _mm256_add_ps(vd_re[row], _mm256_sub_ps(_mm256_mul_ps(mrc_re, vc_re),
                                                                     _mm256_mul_ps(mrc_im, vc_im)));
                vd_im[row] = _mm256_add_ps(vd_im[row], _mm256_add_ps(_mm256_mul_ps(mrc_re, vc_im),
                                                                     _mm256_mul_ps(mrc_im, vc_re)));
            }
            for (size_t m = 0; m < M; m++) {
                _mm256_store_ps(pd[m][0] + v, vd_re[m]);
                _mm256_store_ps(pd[m][1] + v, vd_im[m]);
            }
        }
    }
}


//////////////////////////////////////////////////


/**
 * @brief v0.0.1: version 0.0.1, TensorGrid_CMatrixVector_avx256; double prec
 * 
 * @param dest 
 * @param mat 
 * @param src 
 * @param M 
 * @param N 
 * @param gridSize 
 */
void TensorGrid_CMatrixVector_avx256(double *dest, double *mat, double *src, const size_t M,
                                     const size_t N, const size_t gridSize)
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

                vr_re = _mm256_add_pd(vr_re, _mm256_sub_pd(_mm256_mul_pd(mrc_re, vc_re),
                                                           _mm256_mul_pd(mrc_im, vc_im)));
                vr_im = _mm256_add_pd(vr_im, _mm256_add_pd(_mm256_mul_pd(mrc_re, vc_im),
                                                           _mm256_mul_pd(mrc_im, vc_re)));
            }
            _mm256_store_pd(pd[row][0] + v, vr_re);
            _mm256_store_pd(pd[row][1] + v, vr_im);
        }
    }
}


/**
 * @brief v0.0.1: version 0.0.1, TensorGrid_CMatrixVector_avx256; float prec
 * 
 * @param dest 
 * @param mat 
 * @param src 
 * @param M 
 * @param N 
 * @param gridSize 
 */
void TensorGrid_CMatrixVector_avx256(float *dest, float *mat, float *src, const size_t M,
                                     const size_t N, const size_t gridSize)
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
            for (size_t col = 0; col < N; col++) {
                __m256 mrc_re = _mm256_load_ps(&(pm[row][col][0][v]));
                __m256 mrc_im = _mm256_load_ps(&(pm[row][col][1][v]));
                __m256 vc_re  = _mm256_load_ps(&(ps[col][0][v]));
                __m256 vc_im  = _mm256_load_ps(&(ps[col][1][v]));

                vr_re = _mm256_add_ps(vr_re, _mm256_sub_ps(_mm256_mul_ps(mrc_re, vc_re),
                                                           _mm256_mul_ps(mrc_im, vc_im)));
                vr_im = _mm256_add_ps(vr_im, _mm256_add_ps(_mm256_mul_ps(mrc_re, vc_im),
                                                           _mm256_mul_ps(mrc_im, vc_re)));
            }
            _mm256_store_ps((pd[row][0] + v), vr_re);
            _mm256_store_ps((pd[row][1] + v), vr_im);
        }
    }
}
