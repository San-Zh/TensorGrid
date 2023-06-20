
#pragma once

#include <cstdlib>
#include <complex>
#include "setup.h"
#include "transfer.h"

/// @brief tensorgrid gemv, auto simd
/// @tparam Tp
/// @param dest
/// @param mat
/// @param src
/// @param gridSize
template <typename Tp>
void TensorGrid_CMatrixVector(Tp *dest, Tp *mat, Tp *src, size_t gridSize)
{
    Tp *pd[MAX_ROW][2];
    Tp *ps[MAX_COL][2];
    Tp *pm[MAX_ROW][MAX_COL][2];

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

    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            for (size_t v = 0; v < gridSize; v++) {
                Tp re = pm[row][col][0][v] * ps[col][0][v] - pm[row][col][1][v] * ps[col][1][v];
                Tp im = pm[row][col][0][v] * ps[col][1][v] + pm[row][col][1][v] * ps[col][0][v];
                pd[row][0][v] += re;
                pd[row][1][v] += im;
            }
        }
    }
}

/// @brief
/// @param dest
/// @param X
/// @param Y
/// @param tensorSize
/// @param gridSize
void TensorGrid_CXYpY(DataType *dest, DataType *X, DataType *Y, size_t tensorSize, size_t gridSize)
{
    DataType re, im;
    for (size_t its = 0; its < tensorSize; its++) {
        DataType *Xre = X + its * 2 * gridSize;
        DataType *Xim = X + (its * 2 + 1) * gridSize;
        DataType *Yre = Y + its * 2 * gridSize;
        DataType *Yim = Y + (its * 2 + 1) * gridSize;
        DataType *destre = dest + its * 2 * gridSize;
        DataType *destim = dest + (its * 2 + 1) * gridSize;
        for (size_t v = 0; v < gridSize; v++) {
            re = Xre[v] * Yre[v] - Xim[v] * Yim[v] + Yre[v];
            im = Xre[v] * Yim[v] + Xim[v] * Yre[v] + Yim[v];
            destre[v] = re;
            destim[v] = im;
        }
    }
}
