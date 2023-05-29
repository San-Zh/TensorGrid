
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
    /// the lager grid volume is, the lower performance than the base ComplexAry_MatrixVector()
    // for (size_t row = 0; row < MAX_ROW; row++) {
    //     for (size_t col = 0; col < MAX_COL; col++) {
    //         for (size_t v = 0; v < gridSize; v++) {
    //             dest[v * MAX_ROW + row] += mat[v * MAX_ROW * MAX_COL + row * MAX_COL + col] * src[v * MAX_COL + col];
    //         }
    //     }
    // }
    for (size_t v = 0; v < gridSize; v++) {
        Complex res[MAX_ROW] = {0.0};
        for (size_t row = 0; row < MAX_ROW; row++) {
            for (size_t col = 0; col < MAX_COL; col++) {
                res[row] += mat[v * MAX_ROW * MAX_COL + row * MAX_COL + col] * src[v * MAX_COL + col];
            }
            dest[v * MAX_ROW + row] = res[row];
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

    for (size_t col = 0; col < MAX_COL; col++) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            for (size_t v = 0; v < gridSize; v++) {
                DataType re = pm[row][col][0][v] * ps[col][0][v] - pm[row][col][1][v] * ps[col][1][v];
                DataType im = pm[row][col][0][v] * ps[col][1][v] + pm[row][col][1][v] * ps[col][0][v];
                pd[row][0][v] += re;
                pd[row][1][v] += im;
                // *(pd[row][0] + v) += *(pm[row][col][0] + v) * *(ps[col][0] + v) - *(pm[row][col][1] + v) * *(ps[col][1] + v);
                // *(pd[row][1] + v) += *(pm[row][col][0] + v) * *(ps[col][1] + v) + *(pm[row][col][1] + v) * *(ps[col][0] + v);
            }
        }
    }
}

void TensorGrid_CMatrixVector02(DataType *TGdes, DataType *TGmat, DataType *TGsrc, size_t gridSize)
{
    DataPointer m00re = TGmat + (0 * 2 * MAX_COL + 0) * gridSize;
    DataPointer m00im = TGmat + (0 * 2 * MAX_COL + 1) * gridSize;
    DataPointer m01re = TGmat + (0 * 2 * MAX_COL + 2) * gridSize;
    DataPointer m01im = TGmat + (0 * 2 * MAX_COL + 3) * gridSize;
    DataPointer m02re = TGmat + (0 * 2 * MAX_COL + 4) * gridSize;
    DataPointer m02im = TGmat + (0 * 2 * MAX_COL + 5) * gridSize;
    DataPointer m10re = TGmat + (1 * 2 * MAX_COL + 0) * gridSize;
    DataPointer m10im = TGmat + (1 * 2 * MAX_COL + 1) * gridSize;
    DataPointer m11re = TGmat + (1 * 2 * MAX_COL + 2) * gridSize;
    DataPointer m11im = TGmat + (1 * 2 * MAX_COL + 3) * gridSize;
    DataPointer m12re = TGmat + (1 * 2 * MAX_COL + 4) * gridSize;
    DataPointer m12im = TGmat + (1 * 2 * MAX_COL + 5) * gridSize;
    DataPointer m20re = TGmat + (2 * 2 * MAX_COL + 0) * gridSize;
    DataPointer m20im = TGmat + (2 * 2 * MAX_COL + 1) * gridSize;
    DataPointer m21re = TGmat + (2 * 2 * MAX_COL + 2) * gridSize;
    DataPointer m21im = TGmat + (2 * 2 * MAX_COL + 3) * gridSize;
    DataPointer m22re = TGmat + (2 * 2 * MAX_COL + 4) * gridSize;
    DataPointer m22im = TGmat + (2 * 2 * MAX_COL + 5) * gridSize;

    DataPointer vs0re = TGsrc + 0 * gridSize;
    DataPointer vs0im = TGsrc + 1 * gridSize;
    DataPointer vs1re = TGsrc + 2 * gridSize;
    DataPointer vs1im = TGsrc + 3 * gridSize;
    DataPointer vs2re = TGsrc + 4 * gridSize;
    DataPointer vs2im = TGsrc + 5 * gridSize;

    DataPointer vd0re = TGdes + 0 * gridSize;
    DataPointer vd0im = TGdes + 1 * gridSize;
    DataPointer vd1re = TGdes + 2 * gridSize;
    DataPointer vd1im = TGdes + 3 * gridSize;
    DataPointer vd2re = TGdes + 4 * gridSize;
    DataPointer vd2im = TGdes + 5 * gridSize;

    for (size_t i = 0; i < gridSize; i++) {
        vd0re[i] = m00re[i] * vs0re[i] - m00im[i] * vs0im[i] + m01re[i] * vs1re[i] - m01im[i] * vs1im[i] +
                   m02re[i] * vs2re[i] - m02im[i] * vs2im[i];
        vd0im[i] = m00re[i] * vs0im[i] + m00im[i] * vs0re[i] + m01re[i] * vs1im[i] + m01im[i] * vs1re[i] +
                   m02re[i] * vs2im[i] + m02im[i] * vs2re[i];
        vd1re[i] = m10re[i] * vs0re[i] - m10im[i] * vs0im[i] + m11re[i] * vs1re[i] - m11im[i] * vs1im[i] +
                   m12re[i] * vs2re[i] - m12im[i] * vs2im[i];
        vd1im[i] = m10re[i] * vs0im[i] + m10im[i] * vs0re[i] + m11re[i] * vs1im[i] + m11im[i] * vs1re[i] +
                   m12re[i] * vs2im[i] + m12im[i] * vs2re[i];
        vd2re[i] = m20re[i] * vs0re[i] - m20im[i] * vs0im[i] + m21re[i] * vs1re[i] - m21im[i] * vs1im[i] +
                   m22re[i] * vs2re[i] - m22im[i] * vs2im[i];
        vd2im[i] = m20re[i] * vs0im[i] + m20im[i] * vs0re[i] + m21re[i] * vs1im[i] + m21im[i] * vs1re[i] +
                   m22re[i] * vs2im[i] + m22im[i] * vs2re[i];
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
void TensorGrid_CMatrixVector03(CVectorPtrRow dest, CMatrixPtr mat, CVectorPtrCol src, size_t gridSize)
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