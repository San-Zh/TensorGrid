
#pragma once

#include <cstdlib>
#include <complex>
#include "setup.h"
#include "transfer.h"
#include <cblas.h>

template <typename Tp>
void AryIO(Tp *dest, Tp *src, size_t size)
{
    for (size_t v = 0; v < size; v++) {
        dest[v] = src[v];
    }
}

template <typename Tp>
void AryRead(Tp *src, size_t size)
{
    Tp tmp;
    for (size_t v = 0; v < size; v++) {
        tmp = src[v];
    }
}

template <typename Tp>
void AryWrite(Tp *dest, size_t size)
{
    for (size_t v = 0; v < size; v++) {
        dest[v] = 1.0;
    }
}

template <typename TF>
void ComplexAry_CXYpY(std::complex<TF> *dest, std::complex<TF> *X, std::complex<TF> *Y, size_t size)
{
    for (size_t v = 0; v < size; v++) {
        dest[v] = X[v] * Y[v] + Y[v];
    }
}

/**
 * @brief dest[i] = X[i] * Y[i]
 * 
 * @tparam TF 
 * @param dest 
 * @param X 
 * @param Y 
 * @param tensorSize 
 * @param gridSize 
 */
template <typename TF>
void TensorGrid_CXYpY(TF *dest, TF *X, TF *Y, size_t tensorSize, size_t gridSize)
{
    TF re, im;
    for (size_t its = 0; its < tensorSize; its++) {
        TF *Xre = X + its * 2 * gridSize;
        TF *Xim = X + (its * 2 + 1) * gridSize;
        TF *Yre = Y + its * 2 * gridSize;
        TF *Yim = Y + (its * 2 + 1) * gridSize;
        TF *destre = dest + its * 2 * gridSize;
        TF *destim = dest + (its * 2 + 1) * gridSize;
        for (size_t v = 0; v < gridSize; v++) {
            re = Xre[v] * Yre[v] - Xim[v] * Yim[v] + Yre[v];
            im = Xre[v] * Yim[v] + Xim[v] * Yre[v] + Yim[v];
            destre[v] = re;
            destim[v] = im;
        }
    }
}

template <typename TF>
void ComplexAry_CXTY(std::complex<TF> *dest, std::complex<TF> *X, std::complex<TF> *Y, size_t tensorSize,
                     size_t gridSize)
{
    for (size_t v = 0; v < gridSize; v++) {
        for (size_t its = 0; its < tensorSize; its++) {
            dest[v] += X[its + v * tensorSize] * Y[its + v * tensorSize];
        }
    }
}

template <typename TF>
void TensorGrid_CXTY(TF *dest, TF *X, TF *Y, size_t tensorSize, size_t gridSize)
{
    TF re, im;
    for (size_t its = 0; its < tensorSize; its++) {
        TF *Xre = X + (its * 2) * gridSize;
        TF *Xim = X + (its * 2 + 1) * gridSize;
        TF *Yre = Y + (its * 2) * gridSize;
        TF *Yim = Y + (its * 2 + 1) * gridSize;
        TF *destre = dest;
        TF *destim = dest + gridSize;
        for (size_t v = 0; v < gridSize; v++) {
            re = Xre[v] * Yre[v] - Xim[v] * Yim[v];
            im = Xre[v] * Yim[v] + Xim[v] * Yre[v];
            destre[v] += re;
            destim[v] += im;
        }
    }
}

/// @brief when gridSize > 1024*8 double, this method accelerate rate mothan 4.0, compare to the method as followed;
///         here test pc L1(32K+32K) L2(1024K) L3(17M)
/// @param dest
/// @param mat
/// @param src
/// @param gridSize
template <typename Tp>
void ComplexAry_MatrixVector(std::complex<Tp> *dest, const std::complex<Tp> *mat, const std::complex<Tp> *src,
                             size_t gridSize)
{
#ifdef HAVE_BLAS
    std::complex<Tp> alpha(1, 0), beta(0, 0);
    if (sizeof(Tp) == 4) {
        for (size_t v = 0; v < gridSize; v++) {
            cblas_cgemv(CblasRowMajor, CblasNoTrans, MAX_COL, MAX_ROW, &alpha, &mat[9 * v], 3, &src[3 * v], 1, &beta,
                        &dest[3 * v], 1);
        }
    } else if (sizeof(Tp) == 8) {
        for (size_t v = 0; v < gridSize; v++) {
            cblas_zgemv(CblasRowMajor, CblasNoTrans, MAX_COL, MAX_ROW, &alpha, &mat[9 * v], 3, &src[3 * v], 1, &beta,
                        &dest[3 * v], 1);
        }
    }
#else
    for (size_t v = 0; v < gridSize; v++) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            std::complex<Tp> res = 0.0;
            for (size_t col = 0; col < MAX_COL; col++) {
                // dest[v * MAX_ROW + row] += mat[v * MAX_ROW * MAX_COL + row * MAX_COL + col] * src[v * MAX_COL + col];
                res += mat[v * MAX_ROW * MAX_COL + row * MAX_COL + col] * src[v * MAX_COL + col];
            }
            dest[v * MAX_ROW + row] = res;
        }
    }
#endif
}

template <typename Tp>
void ComplexAry_MatrixVector02(std::complex<Tp> *dest, std::complex<Tp> *mat, std::complex<Tp> *src, size_t gridSize)
{
    for (size_t v = 0; v < gridSize; v++) {
        std::complex<Tp> res[MAX_ROW] = {0.0};
        for (size_t col = 0; col < MAX_COL; col++) {
            for (size_t row = 0; row < MAX_ROW; row++) {
                res[row] += mat[v * MAX_ROW * MAX_COL + row * MAX_COL + col] * src[v * MAX_COL + col];
            }
        }
        for (size_t row = 0; row < MAX_ROW; row++) {
            dest[v * MAX_ROW + row] = res[row];
        }
    }
}

template <typename Tp>
void ComplexAry_MatrixVector_v2(Tp *dest, const Tp *mat, const Tp *src, size_t gridSize)
{
    for (size_t v = 0; v < gridSize; v++) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            Tp res_re = 0.0;
            Tp res_im = 0.0;
            // for (size_t col = 0; col < MAX_COL; col++) {
            //     res_re += mat[v * 2 * MAX_ROW * MAX_COL + row * 2 * MAX_COL + 2 * col + 0] *
            //                   src[v * 2 * MAX_COL + 2 * col + 0] -
            //               mat[v * 2 * MAX_ROW * MAX_COL + row * 2 * MAX_COL + 2 * col + 1] *
            //                   src[v * 2 * MAX_COL + 2 * col + 1];
            //     res_im += mat[v * 2 * MAX_ROW * MAX_COL + row * 2 * MAX_COL + 2 * col + 0] *
            //                   src[v * 2 * MAX_COL + 2 * col + 1] +
            //               mat[v * 2 * MAX_ROW * MAX_COL + row * 2 * MAX_COL + 2 * col + 1] *
            //                   src[v * 2 * MAX_COL + 2 * col + 0];
            // }
            Tp vsrc[MAX_COL][2];
            for (size_t col = 0; col < MAX_COL; col++) {
                vsrc[col][0] = src[v * 2 * MAX_COL + 2 * col + 0];
                vsrc[col][1] = src[v * 2 * MAX_COL + 2 * col + 1];
            }
            for (size_t col = 0; col < MAX_COL; col++) {
                Tp mat_re = mat[v * 2 * MAX_ROW * MAX_COL + row * 2 * MAX_COL + 2 * col + 0];
                Tp mat_im = mat[v * 2 * MAX_ROW * MAX_COL + row * 2 * MAX_COL + 2 * col + 1];
                res_re += mat_re * vsrc[col][0] - mat_im * vsrc[col][1];
                res_im += mat_re * vsrc[col][1] + mat_im * vsrc[col][0];
            }
            dest[v * 2 * MAX_ROW + 2 * row + 0] = res_re;
            dest[v * 2 * MAX_ROW + 2 * row + 1] = res_im;
        }
    }
}

template <typename Tp>
void TensorGrid_CMatrixVector(Tp *dest, Tp *mat, const void *src, size_t gridSize)
{
    Tp *pd[MAX_ROW][2];
    Tp *ps[MAX_COL][2];
    Tp *pm[MAX_ROW][MAX_COL][2];

    for (size_t col = 0; col < MAX_COL; col++) {
        ps[col][0] = (Tp *) src + col * 2 * gridSize;
        ps[col][1] = (Tp *) src + (col * 2 + 1) * gridSize;
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

    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            for (size_t v = 0; v < gridSize; v++) {
                Tp re = pm[row][col][0][v] * ps[col][0][v] - pm[row][col][1][v] * ps[col][1][v];
                Tp im = pm[row][col][0][v] * ps[col][1][v] + pm[row][col][1][v] * ps[col][0][v];
                pd[row][0][v] += re;
                pd[row][1][v] += im;
                // *(pd[row][0] + v) += *(pm[row][col][0] + v) * *(ps[col][0] + v) - *(pm[row][col][1] + v) * *(ps[col][1] + v);
                // *(pd[row][1] + v) += *(pm[row][col][0] + v) * *(ps[col][1] + v) + *(pm[row][col][1] + v) * *(ps[col][0] + v);
            }
        }
    }
}
