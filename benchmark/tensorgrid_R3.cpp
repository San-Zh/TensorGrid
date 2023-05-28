#include <iostream>
#include <cstdlib>
#include <cstring>
#include <complex>
#include "tensorgrid_R3.h"
#include "transfer.h"
#include "timer.h"

#define LT 8
#define LX 8
#define LY 8
#define LZ 8

// #define GRID_VOLUME (LT * LX * LY * LZ)
constexpr size_t GRID_VOLUME()
{
#ifndef SIZE
    return (LT * LX * LY * LZ);
#else
    return SIZE;
#endif
}

void random(DataType *src, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        src[i] = static_cast<DataType>(random()) / RAND_MAX;
        // src[i] = static_cast<DataType>(i);
    }
}

int main(int argc, char **argv)
{
    timer watcher;
    watcher.reset();
    ////// Complxe Array //////
    DataType *mat = new DataType[MAX_ROW * MAX_COL * GRID_VOLUME()];
    DataType *src = new DataType[MAX_COL * GRID_VOLUME()];
    DataType *des = new DataType[MAX_ROW * GRID_VOLUME()];
    /// random
    random(src, MAX_COL * GRID_VOLUME());
    random(mat, MAX_ROW * MAX_COL * GRID_VOLUME());
    memset(des, 0, MAX_ROW * GRID_VOLUME() * sizeof(DataType));

    ////// Tensor Grid //////
    DataType *TGmat = new DataType[MAX_ROW * MAX_COL * GRID_VOLUME()];
    DataType *TGsrc = new DataType[MAX_COL * GRID_VOLUME()];
    DataType *TGdes = new DataType[MAX_ROW * GRID_VOLUME()];

    /// repermute data layout
    tranfer2TG(TGsrc, src, MAX_COL, GRID_VOLUME());
    tranfer2TG(TGdes, des, MAX_ROW, GRID_VOLUME());
    tranfer2TG(TGmat, mat, MAX_ROW * MAX_COL, GRID_VOLUME());
    // memset(TGdes, 0, MAX_ROW * GRID_VOLUME() * sizeof(DataType));
#ifdef Debug
    check_diff(src, TGsrc, MAX_COL, GRID_VOLUME());
    check_diff(mat, TGmat, MAX_ROW * MAX_COL, GRID_VOLUME());
    // check_diff(des, TGdes, MAX_ROW, GRID_VOLUME());
#endif
    /// Tensor Grid pointer
    RVectorPtrRow TGA;
    for (size_t row = 0; row < MAX_ROW; row++) {
        TGA[row] = TGsrc + row * GRID_VOLUME();
    }
    RVectorPtrRow TGB;
    for (size_t col = 0; col < MAX_COL; col++) {
        TGB[col] = TGdes + col * GRID_VOLUME();
    }
    RMatrixPtr TGMAT;
    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            TGMAT[row][col] = TGmat + (row * MAX_ROW + col) * GRID_VOLUME();
        }
    }

    /// TGB = TGMAT * TGA;
    watcher.reset();
    RealArray_MatrixVector(des, mat, src, GRID_VOLUME());
    double timeCMV = watcher.use();

    watcher.reset();
    TensorGrid_RMatrixVector(TGB, TGMAT, TGA, GRID_VOLUME());
    double timeTGMV = watcher.use();

#ifdef Debug
    check_diff(des, TGdes, MAX_ROW, GRID_VOLUME());
#endif

    printf("Gemv: Acc:%6.2lf   RealAry%10.4e TensorGrid %10.4e  GridSize %ld\n", timeCMV / timeTGMV, timeCMV,
           timeTGMV, GRID_VOLUME());
    // fprintf(stderr, "\n===== Total time %g ============  \n\n", watcher.total() );

    delete[] mat;
    delete[] src;
    delete[] des;
    delete[] TGmat;
    delete[] TGsrc;
    delete[] TGdes;

    return 0;
}