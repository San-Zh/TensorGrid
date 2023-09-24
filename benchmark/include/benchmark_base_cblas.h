/**
 * @file benchmark_base_cblas.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <complex>
#include <cblas.h>

/**
 * @brief cblas: zgemv
 * 
 * @param dest 
 * @param mat 
 * @param src 
 * @param M 
 * @param N 
 * @param gridSize 
 */
void ComplexAry_MatrixVector_cblas(const int M, const int N, const std::complex<double> *A,
                                   const std::complex<double> *X, std::complex<double> *Y,
                                   const size_t gridSize)
{
    std::complex<double> alpha(1, 0), beta(0, 0);
    for (size_t v = 0; v < gridSize; v++) {
        cblas_zgemv(CblasRowMajor, CblasNoTrans, M, N, &alpha, &A[M * N * v], N, &X[N * v], 1,
                    &beta, &Y[M * v], 1);
    }
}

/**
 * @brief cblas: cgemv
 */
void ComplexAry_MatrixVector_cblas(const int M, const int N, const std::complex<float> *A,
                                   const std::complex<float> *X, std::complex<float> *Y,
                                   const size_t gridSize)
{
    std::complex<float> alpha(1, 0), beta(0, 0);
    for (size_t v = 0; v < gridSize; v++) {
        cblas_cgemv(CblasRowMajor, CblasNoTrans, M, N, &alpha, &A[M * N * v], N, &X[N * v], 1,
                    &beta, &Y[M * v], 1);
    }
}

/**
 * @brief cblas: cgemm
 * 
 * @param M 
 * @param N 
 * @param K 
 * @param A 
 * @param B 
 * @param C 
 * @param gridSize 
 */
void ComplexAry_MatrixMatrix_cblas(const unsigned M, const unsigned N, const unsigned K,
                                   const float *A, const float *B, float *C, const size_t gridSize)
{
    std::complex<float> alpha(1, 0), beta(0, 0);
    for (size_t v = 0; v < gridSize; v++) {
        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, &alpha, &A[v * 2 * M * K],
                    K, &B[v * 2 * K * N], N, &beta, &C[v * 2 * M * N], N);
    }
}


/**
 * @brief cblas: zgemm
 */
void ComplexAry_MatrixMatrix_cblas(const unsigned M, const unsigned N, const unsigned K,
                                   const double *A, const double *B, double *C,
                                   const size_t gridSize)
{
    std::complex<double> alpha(1, 0), beta(0, 0);
    for (size_t v = 0; v < gridSize; v++) {
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, &alpha, &A[v * 2 * M * K],
                    K, &B[v * 2 * K * N], N, &beta, &C[v * 2 * M * N], N);
    }
}


/**
 * @brief cblas: sgemm
 * 
 * @param M 
 * @param N 
 * @param K 
 * @param A 
 * @param B 
 * @param C 
 * @param gridSize 
 */
void RealAry_MatrixMatrix_cblas(const unsigned M, const unsigned N, const unsigned K,
                                const float *A, const float *B, float *C, const size_t gridSize)
{
    for (size_t v = 0; v < gridSize; v++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, &A[v * M * K], K,
                    &B[v * K * N], N, 0.0, &C[v * M * N], N);
    }
}


/**
 * @brief cblas: dgemm
 */
void RealAry_MatrixMatrix_cblas(const unsigned M, const unsigned N, const unsigned K,
                                const double *A, const double *B, double *C, const size_t gridSize)
{
    for (size_t v = 0; v < gridSize; v++) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, &A[v * M * K], K,
                    &B[v * K * N], N, 0.0, &C[v * M * N], N);
    }
}
