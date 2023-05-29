#include <iostream>
#include <cstdlib>
#include <cstring>
#include <complex>
#include "tensorgrid_R3.h"
#include "transfer.h"
#include "timer.h"

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
#ifdef debug
    check_diff(src, TGsrc, MAX_COL, GRID_VOLUME());
    check_diff(mat, TGmat, MAX_ROW * MAX_COL, GRID_VOLUME());
    // check_diff(des, TGdes, MAX_ROW, GRID_VOLUME());
#endif

    watcher.reset();
    RealArray_MatrixVector(des, mat, src, GRID_VOLUME());
    // RealArray_MatrixVector02(des, mat, src, GRID_VOLUME());
    // TensorGrid_RMatrixVector(TGdes, TGmat, TGsrc, GRID_VOLUME());
    double timeMV = watcher.use();

    // watcher.reset();
    // RealArray_MatrixVector02(des, mat, src, GRID_VOLUME());
    // double timeMV02 = watcher.use();
    // printf("Real Array ACC: timeCMV0 / timeCMV02 = %6.2lf\n", timeMV / timeMV02);

    ////// TGB = TGMAT * TGA;
    watcher.reset();
    TensorGrid_RMatrixVector(TGdes, TGmat, TGsrc, GRID_VOLUME());
    // TensorGrid_RMatrixVector02(TGdes, TGmat, TGsrc, GRID_VOLUME());
    // TensorGrid_RMatrixVector03(TGdes, TGmat, TGsrc, GRID_VOLUME());
    double timeTGMV = watcher.use();

    DataType diff = check_diff(des, TGdes, MAX_ROW, GRID_VOLUME());
    printf("Gemv: Acc:%6.2lf   RealAry%10.2e  TensorGrid %10.2e   diff%10.2e  GridSize %ld\n", timeMV / timeTGMV,
           timeMV, timeTGMV, diff, GRID_VOLUME());

    delete[] mat;
    delete[] src;
    delete[] des;
    delete[] TGmat;
    delete[] TGsrc;
    delete[] TGdes;

    // fprintf(stderr, "\n===== Total time %g ============  \n\n", watcher.total() );

    return 0;
}