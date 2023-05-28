#include <iostream>
#include <cstdlib>
#include <cstring>
#include <complex>
#include "tensorgrid_su3.h"
#include "transfer.h"
#include "timer.h"
#include <omp.h>

#define LT 1024
#define LX 1
#define LY 2
#define LZ 2

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

    // printf("sizeof(DataType) = %ld bytes\n", sizeof(DataType));
    ////// Complxe Array //////
    DataType *mat = new DataType[2 * MAX_ROW * MAX_COL * GRID_VOLUME()];
    DataType *src = new DataType[2 * MAX_COL * GRID_VOLUME()];
    DataType *des = new DataType[2 * MAX_ROW * GRID_VOLUME()];
    DataType *Y = new DataType[2 * MAX_COL * GRID_VOLUME()];
    /// random
    random(src, 2 * MAX_COL * GRID_VOLUME());
    random(mat, 2 * MAX_ROW * MAX_COL * GRID_VOLUME());
    memset(des, 0, 2 * MAX_ROW * GRID_VOLUME() * sizeof(DataType));
    random(Y, 2 * MAX_COL * GRID_VOLUME());
    /// Complex Array
    ComplexPtr CA = (ComplexPtr) (src);
    ComplexPtr CY = (ComplexPtr) (Y);
    ComplexPtr CB = (ComplexPtr) (des);
    ComplexPtr CMat = (ComplexPtr) (mat);

    ////// Tensor Grid //////
    DataType *TGmat = new DataType[2 * MAX_ROW * MAX_COL * GRID_VOLUME()];
    DataType *TGsrc = new DataType[2 * MAX_COL * GRID_VOLUME()];
    DataType *TGdes = new DataType[2 * MAX_ROW * GRID_VOLUME()];
    DataType *TGY = new DataType[2 * MAX_COL * GRID_VOLUME()];

    /// repermute data layout
    tranfer2TG(TGsrc, src, 2 * MAX_COL, GRID_VOLUME());
    tranfer2TG(TGdes, des, 2 * MAX_ROW, GRID_VOLUME());
    tranfer2TG(TGmat, mat, 2 * MAX_ROW * MAX_COL, GRID_VOLUME());
    tranfer2TG(TGY, Y, 2 * MAX_COL, GRID_VOLUME());
    // memset(TGdes, 0, 2 * MAX_ROW * GRID_VOLUME() * sizeof(DataType));
#ifdef Debug
    check_diff(src, TGsrc, 2 * MAX_COL, GRID_VOLUME());
    check_diff(mat, TGmat, 2 * MAX_ROW * MAX_COL, GRID_VOLUME());
    check_diff(Y, TGY, 2 * MAX_COL, GRID_VOLUME());
    // check_diff(des, TGdes, 2 * MAX_ROW, GRID_VOLUME());
#endif
    /// Tensor Grid pointer
    CVectorPtrRow TGA;
    for (size_t row = 0; row < MAX_ROW; row++) {
        TGA[row][0] = TGsrc + (2 * row) * GRID_VOLUME();
        TGA[row][1] = TGsrc + (2 * row + 1) * GRID_VOLUME();
    }
    CVectorPtrRow TGB;
    for (size_t col = 0; col < MAX_COL; col++) {

        TGB[col][0] = TGdes + (2 * col) * GRID_VOLUME();
        TGB[col][1] = TGdes + (2 * col + 1) * GRID_VOLUME();
    }
    CMatrixPtr TGMAT;
    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            TGMAT[row][col][0] = TGmat + (2 * MAX_ROW + 2 * col + 0) * GRID_VOLUME();
            TGMAT[row][col][1] = TGmat + (2 * MAX_ROW + 2 * col + 1) * GRID_VOLUME();
        }
    }

    /// Complex Vector dest = XY + Y;
    {
        watcher.reset();
        ComplexAry_CXYpY(CY, CA, CY, MAX_COL * GRID_VOLUME());
        double timeCV = watcher.use();

        watcher.reset();
        TensorGrid_CXYpY(TGY, TGsrc, TGY, MAX_COL, GRID_VOLUME());
        double timeTGV = watcher.use();

        DataType diffY = check_diff(Y, TGY, 2 * MAX_COL, GRID_VOLUME());
        printf("  XYpY  Acc:%6.2lf    diff%12.4e  GridSize  %ld\n", timeCV / timeTGV, diffY, GRID_VOLUME());
    }

    // { /// TGB = TGMAT * TGA;
    //     watcher.reset();
    //     ComplexAry_MatrixVector(CB, CMat, CA, GRID_VOLUME());
    //     double timeCMV = watcher.use();

    //     watcher.reset();
    //     TensorGrid_CMatrixVector(TGdes, TGmat, TGsrc, GRID_VOLUME());
    //     double timeTGMV = watcher.use();

    //     DataType diffres = check_diff(des, TGdes, 2 * MAX_ROW, GRID_VOLUME());
    //     printf("  Gemv  Acc:%6.2lf   CompAry %10.4e TensorGrid %10.4e  diff%12.4e  GridSize %ld\n", timeCMV / timeTGMV,
    //            timeCMV, timeTGMV, diffres, GRID_VOLUME());
    // }

    fprintf(stderr, " ============ Total time : %g seconds ================\n\n", watcher.total());

    delete[] mat;
    delete[] src;
    delete[] des;
    delete[] TGmat;
    delete[] TGsrc;
    delete[] TGdes;
    delete[] Y;
    delete[] TGY;

    return 0;
}