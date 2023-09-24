/**
 * @file benchmark_TensorGridBlas_autosimd.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-24
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#ifndef MROW
#define MROW 3
#endif

#ifndef NCOL
#define NCOL 3
#endif

template <typename Tp>
void TensorGrid_CMatrixVector(Tp *dest, Tp *mat, const void *src, size_t gridSize)
{
    Tp *pd[MROW][2];
    Tp *ps[NCOL][2];
    Tp *pm[MROW][NCOL][2];

    for (size_t col = 0; col < NCOL; col++) {
        ps[col][0] = (Tp *) src + col * 2 * gridSize;
        ps[col][1] = (Tp *) src + (col * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < MROW; row++) {
        pd[row][0] = dest + row * 2 * gridSize;
        pd[row][1] = dest + (row * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < MROW; row++) {
        for (size_t col = 0; col < NCOL; col++) {
            pm[row][col][0] = mat + (row * 2 * NCOL + 2 * col) * gridSize;
            pm[row][col][1] = mat + (row * 2 * NCOL + 2 * col + 1) * gridSize;
        }
    }

    for (size_t v = 0; v < gridSize; v++) {
        for (size_t row = 0; row < MROW; row++) {
            Tp re = 0.0, im = 0.0;
            for (size_t col = 0; col < NCOL; col++) {
                re += pm[row][col][0][v] * ps[col][0][v] - pm[row][col][1][v] * ps[col][1][v];
                im += pm[row][col][0][v] * ps[col][1][v] + pm[row][col][1][v] * ps[col][0][v];
            }
            pd[row][0][v] = re;
            pd[row][1][v] = im;
        }
    }
}


template <typename Tp>
void TensorGrid_CMatrixVector_Mac(Tp *dest, Tp *mat, const void *src, size_t gridSize)
{
    Tp *pd[MROW][2];
    Tp *ps[NCOL][2];
    Tp *pm[MROW][NCOL][2];

    for (size_t col = 0; col < NCOL; col++) {
        ps[col][0] = (Tp *) src + col * 2 * gridSize;
        ps[col][1] = (Tp *) src + (col * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < MROW; row++) {
        pd[row][0] = dest + row * 2 * gridSize;
        pd[row][1] = dest + (row * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < MROW; row++) {
        for (size_t col = 0; col < NCOL; col++) {
            pm[row][col][0] = mat + (row * 2 * NCOL + 2 * col) * gridSize;
            pm[row][col][1] = mat + (row * 2 * NCOL + 2 * col + 1) * gridSize;
        }
    }

    for (size_t row = 0; row < MROW; row++) {
        for (size_t col = 0; col < NCOL; col++) {
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
        TF *Xre    = X + its * 2 * gridSize;
        TF *Xim    = X + (its * 2 + 1) * gridSize;
        TF *Yre    = Y + its * 2 * gridSize;
        TF *Yim    = Y + (its * 2 + 1) * gridSize;
        TF *destre = dest + its * 2 * gridSize;
        TF *destim = dest + (its * 2 + 1) * gridSize;
        for (size_t v = 0; v < gridSize; v++) {
            re        = Xre[v] * Yre[v] - Xim[v] * Yim[v] + Yre[v];
            im        = Xre[v] * Yim[v] + Xim[v] * Yre[v] + Yim[v];
            destre[v] = re;
            destim[v] = im;
        }
    }
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
void TensorGrid_CXTY(TF *dest, TF *X, TF *Y, size_t tensorSize, size_t gridSize)
{
    TF re, im;
    for (size_t its = 0; its < tensorSize; its++) {
        TF *Xre    = X + (its * 2) * gridSize;
        TF *Xim    = X + (its * 2 + 1) * gridSize;
        TF *Yre    = Y + (its * 2) * gridSize;
        TF *Yim    = Y + (its * 2 + 1) * gridSize;
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
