
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
    size_t Wsmd = 64 / sizeof(DataType);
    size_t Nsmd = sizeGrid / Wsmd;
    size_t v = 0;
    for (size_t i = 0; i < Nsmd; i++) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            for (size_t col = 0; col < MAX_COL; col++) {
                for (size_t vi = 0; vi < Wsmd; vi++) {
                    v = i * Wsmd + vi;
                    dest[v * MAX_ROW + row] +=
                        mat[v * MAX_ROW * MAX_COL + row * MAX_COL + col] * src[v * MAX_COL + col];
                }
            }
        }
    }
}

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
