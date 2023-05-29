
#pragma once

#include <cstdlib>
#include <complex>
#include "setup.h"

template <typename Tp>
void tranfer2TG(Tp *dest, Tp *src, size_t sizeTensor, size_t sizeGrid)
{
    for (size_t v = 0; v < sizeGrid; v++) {
        for (size_t its = 0; its < sizeTensor; its++) {
            *(dest + its * sizeGrid + v) = *(src + v * sizeTensor + its);
        }
    }
}

void random(DataType *src, size_t size)
{
    DataType RdmInv = 1.0 / static_cast<DataType>(RAND_MAX);
    for (size_t i = 0; i < size; i++) {
        src[i] = static_cast<DataType>(random()) * RdmInv;
        // src[i] = static_cast<DataType>(i);
    }
}

template <typename Tp>
Tp check_diff(Tp *A, Tp *B, size_t sizeTensor, size_t sizeGrid)
{
    Tp *pb[sizeTensor];
    for (size_t its = 0; its < sizeTensor; its++) {
        pb[its] = B + its * sizeGrid;
    }

    Tp diff = 0.0;
    Tp sum = 0.0;
    // for (size_t i = 0; i < 4; i++){
    for (size_t i = 0; i < sizeGrid; i++) {
        for (size_t its = 0; its < sizeTensor; its++) {
            diff = A[i * sizeTensor + its] - pb[its][i];
            sum += diff;
#ifdef debug
            printf("%14.4e%14.4e%14.4e\n", A[i * sizeTensor + its], pb[its][i], diff);
#endif
        }
    }
#ifdef debug
    // printf("%42s\n", "--------------------------");
    printf("%28s%14.4e\n\n", "sum of diff", sum);
#endif
    return sum;
}

DataType check_diff(DataType *A, CVectorPtrCol B, size_t sizeGrid)
{
    DataType diffre = 0.0;
    DataType diffim = 0.0;
    DataType sum = 0.0;
    // for (size_t i = 0; i < 4; i++){
    for (size_t i = 0; i < sizeGrid; i++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            diffre = A[i * 2 * MAX_COL + 2 * col + 0] - B[col][0][i];
            diffim = A[i * 2 * MAX_COL + 2 * col + 1] - B[col][1][i];
            sum += diffre + diffim;
#ifdef debug
            printf("%14.4e%14.4e%14.4e\n", A[i * 2 * MAX_COL + 2 * col + 0], B[col][0][i], diffre);
            printf("%14.4e%14.4e%14.4e\n", A[i * 2 * MAX_COL + 2 * col + 1], B[col][1][i], diffim);
#endif
        }
    }
#ifdef debug
    // printf("%42s\n", "--------------------------");
    printf("%28s%14.4e\n\n", "sum of diff", sum);
#endif
    return sum;
}

DataType check_diff(DataType *A, CMatrixPtr B, size_t sizeGrid)
{
    DataType diffre = 0.0;
    DataType diffim = 0.0;
    DataType sum = 0.0;
    // for (size_t i = 0; i < 4; i++){
    for (size_t i = 0; i < sizeGrid; i++) {
        for (size_t row = 0; row < MAX_COL; row++) {
            for (size_t col = 0; col < MAX_COL; col++) {
                diffre = A[i * 2 * MAX_COL * MAX_ROW + row * 2 * MAX_COL + 2 * col + 0] - B[row][col][0][i];
                diffim = A[i * 2 * MAX_COL * MAX_ROW + row * 2 * MAX_COL + 2 * col + 1] - B[row][col][1][i];
                sum += diffre + diffim;
#ifdef debug
                printf("%14.4e%14.4e%14.4e\n", A[i * 2 * MAX_COL * MAX_ROW + row * 2 * MAX_COL + 2 * col + 0],
                       B[row][col][0][i], diffre);
                printf("%14.4e%14.4e%14.4e\n", A[i * 2 * MAX_COL * MAX_ROW + row * 2 * MAX_COL + 2 * col + 1],
                       B[row][col][1][i], diffim);
#endif
            }
        }
    }
#ifdef debug
    printf("%28s%14.4e\n\n", "sum of diff", sum);
    // printf("%42s\n", "--------------------------");
#endif
    return sum;
}