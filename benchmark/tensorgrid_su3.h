
#pragma once

#include <cstdlib>
#include <complex>
#include "setup.h"
#include "transfer.h"

void ComplexAry_CXYpY(ComplexPtr dest, ComplexPtr X, ComplexPtr Y, size_t size)
{
    for (size_t v = 0; v < size; v++) {
        dest[v] = X[v] * Y[v] + Y[v];
    }
}

void TensorGrid_CXYpY(DataType *dest, DataType *X, DataType *Y, size_t tensorSize, size_t gridSize)
{
    double re, im;
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

/// @brief when gridSize > 1024*8 double, this method accelerate rate mothan 4.0, compare to the method as followed;
///         here test pc L1(32K+32K) L2(1024K) L3(17M)
/// @param dest
/// @param mat
/// @param src
/// @param gridSize
void ComplexAry_MatrixVector(ComplexPtr dest, ComplexPtr mat, ComplexPtr src, size_t gridSize)
{
    for (size_t v = 0; v < gridSize; v++) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            for (size_t col = 0; col < MAX_COL; col++) {
                dest[v * MAX_ROW + row] += mat[v * MAX_ROW * MAX_COL + row * MAX_COL + col] * src[v * MAX_COL + col];
            }
        }
    }
}

void ComplexAry_MatrixVector02(ComplexPtr dest, ComplexPtr mat, ComplexPtr src, size_t gridSize)
{
    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            for (size_t v = 0; v < gridSize; v++) {
                dest[v * MAX_ROW + row] += mat[v * MAX_ROW * MAX_COL + row * MAX_COL + col] * src[v * MAX_COL + col];
            }
        }
    }
}

void TensorGrid_CMatrixVector(DataType *dest, DataType *mat, DataType *src, size_t gridSize)
{
    DataType *pd[MAX_ROW][2];
    DataType *ps[MAX_COL][2];
    DataType *pm[MAX_ROW][MAX_COL][2];
    // CVectorPtrRow pd;
    // CVectorPtrRow ps;
    // CMatrixPtr pm;

    for (size_t col = 0; col < MAX_COL; col++) {
        ps[col][0] = src + col * 2 * gridSize;
        ps[col][1] = src + (col * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < MAX_COL; row++) {
        pd[row][0] = dest + row * 2 * gridSize;
        pd[row][1] = dest + (row * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < MAX_COL; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            pm[row][col][0] = mat + (row * 2 * MAX_COL + 2 * col) * gridSize;
            pm[row][col][1] = mat + (row * 2 * MAX_COL + 2 * col + 1) * gridSize;
        }
    }

    DataType re = 0.0;
    DataType im = 0.0;
    for (size_t col = 0; col < MAX_COL; col++) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            for (size_t v = 0; v < gridSize; v++) {
                re = pm[row][col][0][v] * ps[col][0][v] - pm[row][col][1][v] * ps[col][1][v];
                im = pm[row][col][0][v] * ps[col][1][v] + pm[row][col][1][v] * ps[col][0][v];
                pd[row][0][v] += re;
                pd[row][1][v] += im;
                // pd[row][0][v] += pm[row][col][0][v] * ps[col][0][v];
                // pd[row][0][v] -= pm[row][col][1][v] * ps[col][1][v];
                // pd[row][1][v] += pm[row][col][0][v] * ps[col][1][v];
                // pd[row][1][v] += pm[row][col][1][v] * ps[col][0][v];
                // *(pd[row][0] + v) += *(pm[row][col][0] + v) * *(ps[col][0] + v) - *(pm[row][col][1] + v) * *(ps[col][1] + v);
                // *(pd[row][1] + v) += *(pm[row][col][0] + v) * *(ps[col][1] + v) + *(pm[row][col][1] + v) * *(ps[col][0] + v);
            }
        }
    }
}

/**
 * @brief this metod has lower performancee than above
 * 
 * @param dest 
 * @param mat 
 * @param src 
 * @param gridSize 
 */
void TensorGrid_CMatrixVector(CVectorPtrRow dest, CMatrixPtr mat, CVectorPtrCol src, size_t gridSize)
{
    DataType re;
    DataType im;
    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            for (size_t v = 0; v < gridSize; v++) {
                re = mat[row][col][0][v] * src[col][0][v] - mat[row][col][1][v] * src[col][1][v];
                im = mat[row][col][0][v] * src[col][1][v] + mat[row][col][1][v] * src[col][0][v];
                dest[row][0][v] += re;
                dest[row][1][v] += im;
            }
        }
    }
}

#if 0

/**
 * @brief this metod has lower performancee than above
 * 
 * @param dest 
 * @param mat 
 * @param src 
 * @param gridSize 
 */
void TensorGrid_CMatrixVector02(CVectorPtrRow dest, CMatrixPtr mat, CVectorPtrCol src, size_t gridSize)
{
    DataType re;
    DataType im;
    size_t Wsmd = (64 / sizeof(DataType));
    size_t Nsmd = gridSize / Wsmd;
    // printf("Grid size: %ld   Wsmd * Nsmd = %ld * %ld = %ld\n", gridSize, Wsmd, Nsmd, Wsmd * Nsmd);
    for (size_t i = 0; i < Nsmd; i++) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            for (size_t col = 0; col < MAX_COL; col++) {
                // for (size_t v = 0; v < gridSize; v++) {
                for (size_t vi = 0; vi < Wsmd; vi++) {
                    size_t v = i * Wsmd + vi;
                    re = mat[row][col][0][v] * src[col][0][v] - mat[row][col][1][v] * src[col][1][v];
                    im = mat[row][col][0][v] * src[col][1][v] + mat[row][col][1][v] * src[col][0][v];
                    dest[row][0][v] += re;
                    dest[row][1][v] += im;
                }
            }
        }
    }
    // for (size_t row = 0; row < MAX_ROW; row++) {
    //     for (size_t v = 0; v < gridSize; v++) {
    //         dest[row][0][v] += mat[row][0][0][v] * src[0][0][v] - mat[row][0][1][v] * src[0][1][v];
    //         dest[row][1][v] += mat[row][0][0][v] * src[0][1][v] + mat[row][0][1][v] * src[0][0][v];
    //         dest[row][0][v] += mat[row][1][0][v] * src[1][0][v] - mat[row][1][1][v] * src[1][1][v];
    //         dest[row][1][v] += mat[row][1][0][v] * src[1][1][v] + mat[row][1][1][v] * src[1][0][v];
    //         dest[row][0][v] += mat[row][2][0][v] * src[2][0][v] - mat[row][2][1][v] * src[2][1][v];
    //         dest[row][1][v] += mat[row][2][0][v] * src[2][1][v] + mat[row][2][1][v] * src[2][0][v];
    //     }
    // }
#endif