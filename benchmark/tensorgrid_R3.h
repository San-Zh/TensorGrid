
#pragma once

#include <cstdlib>
#include <cstdio>
#include <iostream>

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

typedef DataType *RMatrixPtr[MAX_ROW][MAX_COL];
typedef DataType *RVectorPtrCol[MAX_COL];
typedef DataType *RVectorPtrRow[MAX_ROW];

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

void TensorGrid_RMatrixVector(RVectorPtrRow dest, RMatrixPtr mat, RVectorPtrCol src, size_t gridSize)
{
    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            for (size_t v = 0; v < gridSize; v++) {
                dest[row][v] += mat[row][col][v] * src[col][v];
            }
        }
    }
}
