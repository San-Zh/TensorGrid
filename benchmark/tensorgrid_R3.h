
#pragma once

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include "setup.h"

void RealArray_MatrixVector(DataType *dest, DataType *mat, DataType *src, size_t sizeGrid)
{
    for (size_t v = 0; v < sizeGrid; v++) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            for (size_t col = 0; col < MAX_COL; col++) {
                dest[v * MAX_ROW + row] += mat[v * MAX_ROW * MAX_COL + row * MAX_COL + col] * src[v * MAX_COL + col];
            }
        }
    }
}

void RealArray_MatrixVector02(DataType *dest, DataType *mat, DataType *src, size_t sizeGrid)
{
    DataType res[MAX_ROW] = {0.0, 0.0, 0.0};
    for (size_t v = 0; v < sizeGrid; v++) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            for (size_t col = 0; col < MAX_COL; col++) {
                res[row] += mat[v * MAX_ROW * MAX_COL + row * MAX_COL + col] * src[v * MAX_COL + col];
            }
            dest[v * MAX_ROW + row] = res[row];

            // dest[v * MAX_ROW + row] = mat[v * MAX_ROW * MAX_COL + row * MAX_COL + 0] * src[v * MAX_COL + 0] +
            //                           mat[v * MAX_ROW * MAX_COL + row * MAX_COL + 1] * src[v * MAX_COL + 1] +
            //                           mat[v * MAX_ROW * MAX_COL + row * MAX_COL + 2] * src[v * MAX_COL + 2];
        }
    }
}

/**
 * @brief TensorGrid base
 * 
 * @param TGdes 
 * @param TGmat 
 * @param TGsrc 
 * @param gridSize 
 */
void TensorGrid_RMatrixVector(DataType *TGdes, DataType *TGmat, DataType *TGsrc, size_t gridSize)
{
    RVectorPtrRow TGX;
    for (size_t col = 0; col < MAX_COL; col++) {
        TGX[col] = TGsrc + col * GRID_VOLUME();
    }
    RVectorPtrRow TGY;
    for (size_t row = 0; row < MAX_ROW; row++) {
        TGY[row] = TGdes + row * GRID_VOLUME();
    }
    RMatrixPtr TGMAT;
    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            TGMAT[row][col] = TGmat + (row * MAX_ROW + col) * GRID_VOLUME();
        }
    }
    // DataType res[MAX_ROW] = {0.0, 0.0, 0.0};
    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            for (size_t v = 0; v < gridSize; v++) {
                TGY[row][v] += TGMAT[row][col][v] * TGX[col][v];
            }
        }
    }
}

void TensorGrid_RMatrixVector02(DataType *TGdes, DataType *TGmat, DataType *TGsrc, size_t gridSize)
{
    DataPointer m00 = TGmat + (0 * MAX_COL + 0) * gridSize;
    DataPointer m01 = TGmat + (0 * MAX_COL + 1) * gridSize;
    DataPointer m02 = TGmat + (0 * MAX_COL + 2) * gridSize;
    DataPointer m10 = TGmat + (1 * MAX_COL + 0) * gridSize;
    DataPointer m11 = TGmat + (1 * MAX_COL + 1) * gridSize;
    DataPointer m12 = TGmat + (1 * MAX_COL + 2) * gridSize;
    DataPointer m20 = TGmat + (2 * MAX_COL + 0) * gridSize;
    DataPointer m21 = TGmat + (2 * MAX_COL + 1) * gridSize;
    DataPointer m22 = TGmat + (2 * MAX_COL + 2) * gridSize;

    DataPointer vs0 = TGsrc + 0 * gridSize;
    DataPointer vs1 = TGsrc + 1 * gridSize;
    DataPointer vs2 = TGsrc + 2 * gridSize;

    DataPointer vd0 = TGdes + 0 * gridSize;
    DataPointer vd1 = TGdes + 1 * gridSize;
    DataPointer vd2 = TGdes + 2 * gridSize;

    for (size_t i = 0; i < gridSize; i++) {
        // vd0[i] = m00[i] * vs0[i] + m01[i] * vs1[i] + m02[i] * vs2[i];
        // vd1[i] = m10[i] * vs0[i] + m11[i] * vs1[i] + m12[i] * vs2[i];
        // vd2[i] = m20[i] * vs0[i] + m21[i] * vs1[i] + m22[i] * vs2[i];

        // vd0[i] = m00[i] * vs0[i], vd0[i] += m01[i] * vs1[i], vd0[i] += m02[i] * vs2[i];
        // vd1[i] = m10[i] * vs0[i], vd1[i] += m11[i] * vs1[i], vd1[i] += m12[i] * vs2[i];
        // vd2[i] = m20[i] * vs0[i], vd2[i] += m21[i] * vs1[i], vd2[i] += m22[i] * vs2[i];

        *(vd0) = *(m00) * *(vs0) + *(m01) * *(vs1) + *(m02) * *(vs2);
        *(vd1) = *(m10) * *(vs0) + *(m11) * *(vs1) + *(m12) * *(vs2);
        *(vd2) = *(m20) * *(vs0) + *(m21) * *(vs1) + *(m22) * *(vs2);
        vs0++, vs1++, vs2++;
        m00++, m01++, m02++;
        m10++, m11++, m12++;
        m20++, m21++, m22++;
        vd0++, vd1++, vd2++;
    }
}

void TensorGrid_RMatrixVector03(DataType *TGdes, DataType *TGmat, DataType *TGsrc, size_t gridSize)
{
    RVectorPtrRow TGX;
    for (size_t col = 0; col < MAX_COL; col++) {
        TGX[col] = TGsrc + col * GRID_VOLUME();
    }
    RVectorPtrRow TGY;
    for (size_t row = 0; row < MAX_ROW; row++) {
        TGY[row] = TGdes + row * GRID_VOLUME();
    }
    RMatrixPtr TGMAT;
    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            TGMAT[row][col] = TGmat + (row * MAX_ROW + col) * GRID_VOLUME();
        }
    }
    size_t Wsmd = 64 / sizeof(DataType);
    size_t Nsmd = gridSize / Wsmd;
    size_t v = 0;
    for (size_t i = 0; i < Nsmd; i++) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            for (size_t col = 0; col < MAX_COL; col++) {
                for (size_t vi = 0; vi < Wsmd; vi++) {
                    v = i * Wsmd + vi;
                    TGY[row][v] += TGMAT[row][col][v] * TGX[col][v];
                }
            }
        }
    }
}
