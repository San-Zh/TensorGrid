
#pragma once

#include <cstdlib>
#include <complex>

#ifndef MAX_ROW
#define MAX_ROW 3
#endif

#ifndef MAX_COL
#define MAX_COL 3
#endif

#ifndef PRECISION
#define PRECISION DOUBLE
#define DataType double
#else
#define PRECISION SINGLE
#define DataType float
#endif

typedef DataType *CMatrixPtr[MAX_ROW][MAX_COL][2];
typedef DataType *CVectorPtrCol[MAX_COL][2];
typedef DataType *CVectorPtrRow[MAX_ROW][2];
typedef std::complex<DataType> *ComplexPtr;

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

void ComplexAry_MatrixVector(ComplexPtr dest, ComplexPtr mat, ComplexPtr src, size_t gridSize)
{
    for (size_t v = 0; v < gridSize; v++) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            for (size_t col = 0; col < MAX_COL; col++) {
                dest[v * MAX_ROW + row] += mat[v * MAX_ROW * MAX_COL + row * MAX_COL + col] * src[v * MAX_COL + col];
            }
        }
    }
    // ComplexPtr pdes = dest;
    // ComplexPtr pmat = mat;
    // ComplexPtr psrc = src;
    // for (size_t v = 0; v < gridSize; v++)
    // {
    //     pdes += v * MAX_ROW;
    //     psrc += v * MAX_COL;
    //     pmat += v * MAX_ROW * MAX_COL;

    //     for (size_t row = 0; row < MAX_ROW; row++)
    //     {
    //         for (size_t col = 0; col < MAX_COL; col++)
    //         {
    //             pdes[row] += pmat[row * MAX_COL + col] * psrc[col];
    //             printf("%ld  %ld  %ld OK\n", v, row, col);
    //         }
    //     }
    // }
}

void TensorGrid_CMatrixVector(CVectorPtrRow dest, CMatrixPtr mat, CVectorPtrCol src, size_t gridSize)
{
    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            for (size_t v = 0; v < gridSize; v++) {
                dest[row][0][v] += mat[row][col][0][v] * src[col][0][v] - mat[row][col][1][v] * src[col][1][v];
                dest[row][1][v] += mat[row][col][0][v] * src[col][1][v] + mat[row][col][1][v] * src[col][0][v];
            }
        }
    }
}

void TensorGrid_CMatrixVector(DataType *dest, DataType *mat, DataType *src, size_t gridSize)
{
    DataType *ps[MAX_ROW][2];
    DataType *pd[MAX_COL][2];
    DataType *pm[MAX_ROW][MAX_COL][2];

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

    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            for (size_t v = 0; v < gridSize; v++) {
                // pd[row][0][v] += pm[row][col][0][v] * ps[col][0][v] - pm[row][col][1][v] * ps[col][1][v];
                // pd[row][1][v] += pm[row][col][0][v] * ps[col][1][v] + pm[row][col][1][v] * ps[col][0][v];
                // pd[row][0][v] += pm[row][col][0][v] * ps[col][0][v];
                // pd[row][0][v] -= pm[row][col][1][v] * ps[col][1][v];
                // pd[row][1][v] += pm[row][col][0][v] * ps[col][1][v];
                // pd[row][1][v] += pm[row][col][1][v] * ps[col][0][v];
                *(pd[row][0] + v) += *(pm[row][col][0] + v) * *(ps[col][0] + v) - *(pm[row][col][1] + v) * *(ps[col][1] + v);
                *(pd[row][1] + v) += *(pm[row][col][0] + v) * *(ps[col][1] + v) + *(pm[row][col][1] + v) * *(ps[col][0] + v);
            }
        }
    }
}
