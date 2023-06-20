
#pragma once

#include <cstdlib>
#include <complex>
#include "setup.h"
#include "transfer.h"

/// @brief complex vector, dest := X * Y + Y
/// @param dest 
/// @param X 
/// @param Y 
/// @param size 
template<typename Tp>
void ComplexAry_CXYpY(std::complex<Tp> dest, std::complex<Tp> X, std::complex<Tp> Y, size_t size)
{
    for (size_t v = 0; v < size; v++) {
        dest[v] = X[v] * Y[v] + Y[v];
    }
}

/// @brief complex array, gemv()
/// @param dest 
/// @param X 
/// @param Y 
/// @param tensorSize 
/// @param gridSize 
template<typename Tp>
void ComplexAry_CXTY(std::complex<Tp> dest, std::complex<Tp> X, std::complex<Tp> Y, size_t tensorSize, size_t gridSize)
{
    for (size_t v = 0; v < gridSize; v++) {
        for (size_t its = 0; its < tensorSize; its++) {
            dest[v] += X[its + v * tensorSize] * Y[its + v * tensorSize];
        }
    }
}


/// @brief 
/// @tparam Tp 
/// @param dest 
/// @param mat 
/// @param src 
/// @param gridSize 
template <typename Tp>
void ComplexAry_MatrixVector(std::complex<Tp> *dest, std::complex<Tp> *mat, std::complex<Tp> *src, size_t gridSize)
{
    for (size_t v = 0; v < gridSize; v++) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            for (size_t col = 0; col < MAX_COL; col++) {
                dest[v * MAX_ROW + row] += mat[v * MAX_ROW * MAX_COL + row * MAX_COL + col] * src[v * MAX_COL + col];
            }
        }
    }
}


