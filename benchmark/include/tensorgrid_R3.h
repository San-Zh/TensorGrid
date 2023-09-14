/**
 * @file tensorgrid_R3.h
 * @author your name (you@domain.com)
 * @brief \todo TO BE MERGED
 * @version 0.1
 * @date 2023-09-14
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once

#include <iostream>

template <typename Tp>
void RealArray_RXYpY(Tp *dest, Tp *X, Tp *Y, size_t size)
{ 
    for (size_t v = 0; v < size; v++) {
        dest[v] = X[v] * Y[v] + Y[v];
    }
}


template <typename Tp>
void RealArray_MatrixVector(Tp *dest, Tp *mat, Tp *src, size_t sizeGrid)
{
    for (size_t v = 0; v < sizeGrid; v++) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            for (size_t col = 0; col < NCOL; col++) {
                dest[v * MAX_ROW + row] += mat[v * MAX_ROW * NCOL + row * NCOL + col] * src[v * NCOL + col];
            }
        }
    }
}

template <typename Tp>
void RealArray_MatrixVector02(Tp *dest, Tp *mat, Tp *src, size_t sizeGrid)
{
    Tp res[MAX_ROW] = {0.0, 0.0, 0.0};
    for (size_t v = 0; v < sizeGrid; v++) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            for (size_t col = 0; col < NCOL; col++) {
                res[row] += mat[v * MAX_ROW * NCOL + row * NCOL + col] * src[v * NCOL + col];
            }
            dest[v * MAX_ROW + row] = res[row];

            // dest[v * MAX_ROW + row] = mat[v * MAX_ROW * NCOL + row * NCOL + 0] * src[v * NCOL + 0] +
            //                           mat[v * MAX_ROW * NCOL + row * NCOL + 1] * src[v * NCOL + 1] +
            //                           mat[v * MAX_ROW * NCOL + row * NCOL + 2] * src[v * NCOL + 2];
        }
    }
}


template <typename Tp>
void TensorGrid_RXYpY(Tp *dest, Tp *X, Tp *Y, size_t tensorSize, size_t gridSize)
{
    for (size_t its = 0; its < tensorSize; its++) {
        Tp *Xre    = X + its * gridSize;
        Tp *Yre    = Y + its * gridSize;
        Tp *destre = dest + its * gridSize;
        for (size_t v = 0; v < gridSize; v++) { destre[v] = Xre[v] * Yre[v] + Yre[v]; }
    }
}


template<typename Tp>
void TensorGrid_RMatrixVector(Tp *TGdes, Tp *TGmat, Tp *TGsrc, size_t gridSize)
{
    RVectorPtrRow TGX;
    for (size_t col = 0; col < NCOL; col++) {
        TGX[col] = TGsrc + col * GRID_VOLUME();
    }
    RVectorPtrRow TGY;
    for (size_t row = 0; row < MAX_ROW; row++) {
        TGY[row] = TGdes + row * GRID_VOLUME();
    }
    RMatrixPtr TGMAT;
    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < NCOL; col++) {
            TGMAT[row][col] = TGmat + (row * MAX_ROW + col) * GRID_VOLUME();
        }
    }
    // Tp res[MAX_ROW] = {0.0, 0.0, 0.0};
    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < NCOL; col++) {
            for (size_t v = 0; v < gridSize; v++) {
                TGY[row][v] += TGMAT[row][col][v] * TGX[col][v];
            }
        }
    }
}