/**
 * @file benchmark_base_generic.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-16
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include <complex>

#if defined(HAVE_BLAS)
#include <cblas.h>
#endif



/**
 * @brief  when gridSize > 1024*8 double, this method accelerate rate mothan 4.0, 
 * compare to  method as followed; here test pc L1(32K+32K) L2(1024K) L3(17M)
 * 
 * @tparam Tp 
 * @param dest 
 * @param mat 
 * @param src 
 * @param gridSize 
 */
template <int M, int N, typename Tp>
void ComplexAry_MatrixVector_Mac(std::complex<Tp> *&dest, const std::complex<Tp> *&mat,
                                 const std::complex<Tp> *&src, const int gridSize)
{
#ifdef HAVE_BLAS
    std::complex<Tp> alpha(1, 0), beta(0, 0);
    if (sizeof(Tp) == 4) {
        for (size_t v = 0; v < gridSize; v++) {
            cblas_cgemv(CblasRowMajor, CblasNoTrans, N, M, &alpha, &mat[9 * v], 3, &src[3 * v], 1,
                        &beta, &dest[3 * v], 1);
        }
    } else if (sizeof(Tp) == 8) {
        for (size_t v = 0; v < gridSize; v++) {
            cblas_zgemv(CblasRowMajor, CblasNoTrans, N, M, &alpha, &mat[9 * v], 3, &src[3 * v], 1,
                        &beta, &dest[3 * v], 1);
        }
    }
#else
    for (size_t v = 0; v < gridSize; v++) {
        for (size_t row = 0; row < M; row++) {
            std::complex<Tp> res = 0.0;
            for (size_t col = 0; col < N; col++) {
                dest[v * M + row] += mat[v * M * N + row * N + col] * src[v * N + col];
                // res += mat[v * M * N + row * N + col] * src[v * N + col];
            }
            // dest[v * M + row] = res;
        }
    }
#endif
}


/**
 * @brief  when gridSize > 1024*8 double, this method accelerate rate mothan 4.0, 
 * compare to  method as followed; here test pc L1(32K+32K) L2(1024K) L3(17M)
 * 
 * @tparam Tp 
 * @param dest 
 * @param mat 
 * @param src 
 * @param gridSize 
 */
template <int M, int N, typename Tp>
void ComplexAry_MatrixVector(std::complex<Tp> *dest, const std::complex<Tp> *mat,
                             const std::complex<Tp> *src, const int gridSize)
{
#ifdef HAVE_BLAS
    std::complex<Tp> alpha(1, 0), beta(0, 0);
    if (sizeof(Tp) == 4) {
        for (size_t v = 0; v < gridSize; v++) {
            cblas_cgemv(CblasRowMajor, CblasNoTrans, N, M, &alpha, &mat[9 * v], 3, &src[3 * v], 1,
                        &beta, &dest[3 * v], 1);
        }
    } else if (sizeof(Tp) == 8) {
        for (size_t v = 0; v < gridSize; v++) {
            cblas_zgemv(CblasRowMajor, CblasNoTrans, N, M, &alpha, &mat[9 * v], 3, &src[3 * v], 1,
                        &beta, &dest[3 * v], 1);
        }
    }
#else
    for (size_t v = 0; v < gridSize; v++) {
        for (size_t row = 0; row < M; row++) {
            std::complex<Tp> res = 0.0;
            for (size_t col = 0; col < N; col++) {
                // dest[v * M + row] += mat[v * M * N + row * N + col] * src[v * N + col];
                res += mat[v * M * N + row * N + col] * src[v * N + col];
            }
            dest[v * M + row] = res;
        }
    }
#endif
}

template <int M, int N, typename Tp>
void ComplexAry_MatrixVector02(std::complex<Tp> *dest, std::complex<Tp> *mat, std::complex<Tp> *src,
                               const int gridSize)
{
    for (size_t v = 0; v < gridSize; v++) {
        std::complex<Tp> res[M] = {0.0};
        for (size_t col = 0; col < N; col++) {
            for (size_t row = 0; row < M; row++) {
                res[row] += mat[v * M * N + row * N + col] * src[v * N + col];
            }
        }
        for (size_t row = 0; row < M; row++) { dest[v * M + row] = res[row]; }
    }
}

template <int M, int N, typename Tp>
void ComplexAry_MatrixVector_v2(Tp *dest, const Tp *mat, const Tp *src, const int gridSize)
{
    for (size_t v = 0; v < gridSize; v++) {
        for (size_t row = 0; row < M; row++) {
            Tp res_re = 0.0;
            Tp res_im = 0.0;
            Tp vsrc[N][2];
            for (size_t col = 0; col < N; col++) {
                vsrc[col][0] = src[v * 2 * N + 2 * col + 0];
                vsrc[col][1] = src[v * 2 * N + 2 * col + 1];
            }
            for (size_t col = 0; col < N; col++) {
                Tp mat_re = mat[v * 2 * M * N + row * 2 * N + 2 * col + 0];
                Tp mat_im = mat[v * 2 * M * N + row * 2 * N + 2 * col + 1];
                res_re += mat_re * vsrc[col][0] - mat_im * vsrc[col][1];
                res_im += mat_re * vsrc[col][1] + mat_im * vsrc[col][0];
            }
            dest[v * 2 * M + 2 * row + 0] = res_re;
            dest[v * 2 * M + 2 * row + 1] = res_im;
        }
    }
}

template <typename Tp>
Tp normX(Tp *const &_X, size_t const &_N)
{
    size_t cnt = _N;
    Tp     res = 0.0;
    while ((cnt--)) { res += _X[cnt] * _X[cnt]; }
    return res;
}

// clang-format off

/**
 * @brief 
 * 
 * @tparam Tp 
 * @param dest 
 * @param src 
 * @param size 
 */
template <typename Tp>
void AryIO(Tp *dest, Tp *src, size_t size) { for (size_t v = 0; v < size; v++) { dest[v] = src[v]; } }


/**
 * @brief 
 * 
 * @tparam Tp 
 * @param src 
 * @param size 
 */
template <typename Tp>
void AryRead(Tp *src, size_t size) { Tp tmp; for (size_t v = 0; v < size; v++) { tmp = src[v]; } }


/**
 * @brief 
 * 
 * @tparam Tp 
 * @param dest 
 * @param size 
 */
template <typename Tp>
void AryWrite(Tp *dest, size_t size) { for (size_t v = 0; v < size; v++) { dest[v] = 1.0; } }


/**
 * @brief 
 * 
 * @tparam TF 
 * @param dest 
 * @param X 
 * @param Y 
 * @param size 
 */
template <typename TF>
void ComplexAry_CXYpY(std::complex<TF> *dest, std::complex<TF> *X, std::complex<TF> *Y, size_t size)
{
    for (size_t v = 0; v < size; v++) { dest[v] = X[v] * Y[v] + Y[v]; }
}


/**
 * @brief 
 * 
 * @tparam TF 
 * @param dest 
 * @param X 
 * @param Y 
 * @param tensorSize 
 * @param gridSize 
 */
template <typename TF>
void ComplexAry_CXdotY(std::complex<TF> *dest, std::complex<TF> *X, std::complex<TF> *Y, size_t tensorSize, const int gridSize)
{
    for (size_t v = 0; v < gridSize; v++) {
        for (size_t its = 0; its < tensorSize; its++) {
            dest[v] += X[its + v * tensorSize] * Y[its + v * tensorSize];
        }
    }
}
