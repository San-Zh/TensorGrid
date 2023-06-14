
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>

#define GridSize 1024
#define MAX_ROW 3
#define MAX_COL 3
#define DataType float

int main(int argc, char **argv)
{
    ////// Tensor Grid //////
    DataType *mat = new DataType[2 * MAX_ROW * MAX_COL * GridSize];
    DataType *src = new DataType[2 * MAX_COL * GridSize];
    DataType *dest = new DataType[2 * MAX_ROW * GridSize];

    DataType *pd[MAX_ROW][2];
    DataType *ps[MAX_COL][2];
    DataType *pm[MAX_ROW][MAX_COL][2];
    // CVectorPtrRow pd;
    // CVectorPtrRow ps;
    // CMatrixPtr pm;

    for (size_t col = 0; col < MAX_COL; col++) {
        ps[col][0] = src + col * 2 * GridSize;
        ps[col][1] = src + (col * 2 + 1) * GridSize;
    }
    for (size_t row = 0; row < MAX_ROW; row++) {
        pd[row][0] = dest + row * 2 * GridSize;
        pd[row][1] = dest + (row * 2 + 1) * GridSize;
    }
    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            pm[row][col][0] = mat + (row * 2 * MAX_COL + 2 * col) * GridSize;
            pm[row][col][1] = mat + (row * 2 * MAX_COL + 2 * col + 1) * GridSize;
        }
    }

    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            for (size_t v = 0; v < GridSize; v++) {
                DataType re = pm[row][col][0][v] * ps[col][0][v] - pm[row][col][1][v] * ps[col][1][v];
                DataType im = pm[row][col][0][v] * ps[col][1][v] + pm[row][col][1][v] * ps[col][0][v];
                pd[row][0][v] += re;
                pd[row][1][v] += im;
            }
        }
    }

    return 0;
}