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
#include "benchmark_base_cblas.h"
#endif



/**
 * @brief  when gridSize > 1024*8 double, this method accelerate rate mothan 4.0, 
 * compare to  method as followed; here test pc L1(32K+32K) L2(1024K) L3(17M)
 * 
 * @tparam Tp 
 * @param A 
 * @param X 
 * @param Y 
 * @param gridSize 
 */
template <typename Tp>
void ComplexAry_MatrixVector_Mac(const unsigned M, const unsigned N, const Tp *A, const Tp *X,
                                 Tp *Y, const int gridSize)
{
    for (size_t v = 0; v < gridSize; v++) {
        for (size_t row = 0; row < M; row++) {
            std::complex<Tp> res = 0.0;
            for (size_t col = 0; col < N; col++) {
                Y[v * M + row] += A[v * M * N + row * N + col] * X[v * N + col];
            }
        }
    }
}


/**
 * @brief Y(i)[m] = sum_n A(i)[m][n] * X(i)[n];
 * 
 * @tparam Tp 
 * @param M 
 * @param N 
 * @param A 
 * @param X 
 * @param Y 
 * @param gridSize 
 */
template <typename Tp>
void AOS_Gemv_batch(const unsigned M, const unsigned N, const Tp *A, const Tp *X, Tp *Y,
                    const int gridSize)
{
    for (size_t v = 0; v < gridSize; v++) {
        for (size_t row = 0; row < M; row++) {
            Tp res = 0.0;
            for (size_t col = 0; col < N; col++) {
                res += A[v * M * N + row * N + col] * X[v * N + col];
            }
            Y[v * M + row] = res;
        }
    }
}


/**
 * @brief C(i)[m][n] = sum_n A(i)[m][k] * B(i)[k][n]
 * 
 * @tparam Tp 
 * @param M 
 * @param N 
 * @param K 
 * @param A 
 * @param B 
 * @param C 
 * @param gridSize 
 */
template <typename Tp>
void AOS_Gemm_batch(const unsigned M, const unsigned N, const unsigned K, const Tp *A, const Tp *B,
                    Tp *C, const size_t gridSize)
{
    for (size_t v = 0; v < gridSize; v++) {
        Tp Ret[M][N];
        for (unsigned m = 0; m < M; m++) {
            for (unsigned n = 0; n < N; n++) {
                Ret[m][n] = Tp(0.0);
                for (unsigned k = 0; k < K; k++) {
                    Ret[m][n] += A[v * M * K + m * K + k] * B[v * K * N + k * N + n];
                }
                C[v * M * N + m * N + n] = Ret[m][n];
            }
        }
    }
    // {
    //     const size_t vch = 245;
    //     std::cout << "C[" << M << "][" << N << "] = " << std::endl;
    //     for (unsigned m = 0; m < M; m++) {
    //         for (unsigned n = 0; n < N; n++) { std::cout << " " << C[vch * M * N + m * N + n]; }
    //         std::cout << std::endl;
    //     }
    //     std::cout << "-----------------------" << std::endl;
    // }
}



template <typename Tp>
void AOS_Gemv_batch_v1(const unsigned M, const unsigned N, std::complex<Tp> *A, std::complex<Tp> *X,
                       std::complex<Tp> *Y, const int gridSize)
{
    for (size_t v = 0; v < gridSize; v++) {
        std::complex<Tp> res[M];
        for (size_t col = 0; col < N; col++) {
            for (size_t row = 0; row < M; row++) {
                res[row] += A[v * M * N + row * N + col] * X[v * N + col];
            }
        }
        for (size_t row = 0; row < M; row++) { Y[v * M + row] = res[row]; }
    }
}

template <typename Tp>
void AOS_Gemv_batch_v2(const unsigned M, const unsigned N, const Tp *A, const Tp *X, Tp *Y,
                        const int gridSize)
{
    for (size_t v = 0; v < gridSize; v++) {
        for (size_t row = 0; row < M; row++) {
            Tp res_re = 0.0;
            Tp res_im = 0.0;
            Tp vsrc[N][2];
            for (size_t col = 0; col < N; col++) {
                vsrc[col][0] = X[v * 2 * N + 2 * col + 0];
                vsrc[col][1] = X[v * 2 * N + 2 * col + 1];
            }
            for (size_t col = 0; col < N; col++) {
                Tp mat_re = A[v * 2 * M * N + row * 2 * N + 2 * col + 0];
                Tp mat_im = A[v * 2 * M * N + row * 2 * N + 2 * col + 1];
                res_re += mat_re * vsrc[col][0] - mat_im * vsrc[col][1];
                res_im += mat_re * vsrc[col][1] + mat_im * vsrc[col][0];
            }
            Y[v * 2 * M + 2 * row + 0] = res_re;
            Y[v * 2 * M + 2 * row + 1] = res_im;
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
 * @param Y 
 * @param X 
 * @param size 
 */
template <typename Tp>
void AryIO(Tp *Y, Tp *X, size_t size) { for (size_t v = 0; v < size; v++) { Y[v] = X[v]; } }


/**
 * @brief 
 * 
 * @tparam Tp 
 * @param X 
 * @param size 
 */
template <typename Tp>
void AryRead(Tp *X, size_t size) { Tp tmp; for (size_t v = 0; v < size; v++) { tmp = X[v]; } }


/**
 * @brief 
 * 
 * @tparam Tp 
 * @param Y 
 * @param size 
 */
template <typename Tp>
void AryWrite(Tp *Y, size_t size) { for (size_t v = 0; v < size; v++) { Y[v] = 1.0; } }


/**
 * @brief 
 * 
 * @tparam TF 
 * @param des 
 * @param X 
 * @param Y 
 * @param size 
 */
template <typename TF>
void ComplexAry_CXYpY(std::complex<TF> *des, std::complex<TF> *X, std::complex<TF> *Y, size_t size)
{
    for (size_t v = 0; v < size; v++) { des[v] = X[v] * Y[v] + Y[v]; }
}


/**
 * @brief 
 * 
 * @tparam TF 
 * @param Y 
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
