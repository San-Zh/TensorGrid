
#include <omp.h>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "tensorgrid_su3.h"
#include "timer.h"
#include "transfer.h"

int main(int argc, char **argv)
{
    timer watcher;
    watcher.reset();

    // printf("sizeof(DataType) = %ld bytes\n", sizeof(DataType));
    ////// Complxe Array //////
    DataType *mat = new DataType[2 * MAX_ROW * MAX_COL * GRID_VOLUME()];
    DataType *src = new DataType[2 * MAX_COL * GRID_VOLUME()];
    DataType *des = new DataType[2 * MAX_ROW * GRID_VOLUME()];
    /// random
    random(src, 2 * MAX_COL * GRID_VOLUME());
    random(mat, 2 * MAX_ROW * MAX_COL * GRID_VOLUME());
    memset(des, 0, 2 * MAX_ROW * GRID_VOLUME() * sizeof(DataType));
    /// Complex Array
    ComplexPtr CMat = (ComplexPtr)(mat);
    ComplexPtr Csrc = (ComplexPtr)(src);
    ComplexPtr Cdes = (ComplexPtr)(des);

    ////// Tensor Grid //////
    DataType *TGmat = new DataType[2 * MAX_ROW * MAX_COL * GRID_VOLUME()];
    DataType *TGsrc = new DataType[2 * MAX_COL * GRID_VOLUME()];
    DataType *TGdes = new DataType[2 * MAX_ROW * GRID_VOLUME()];

    /// repermute data layout
    watcher.reset();
    tranfer2TG(TGsrc, src, 2 * MAX_COL, GRID_VOLUME());
    tranfer2TG(TGmat, mat, 2 * MAX_ROW * MAX_COL, GRID_VOLUME());
    tranfer2TG(TGdes, des, 2 * MAX_ROW, GRID_VOLUME());
    double timeTrans = watcher.use();
    // tranfer2general(des, TGdes, 2 * MAX_ROW, GRID_VOLUME());
    // printf("transfer time: %10.2e\n", timeTrans);

    // memset(TGdes, 0, 2 * MAX_ROW * GRID_VOLUME() * sizeof(DataType));
#ifdef debug
    diff_Ary_TGAry(src, TGsrc, 2 * MAX_COL, GRID_VOLUME());
    diff_Ary_TGAry(mat, TGmat, 2 * MAX_ROW * MAX_COL, GRID_VOLUME());
    // diff_Ary_TGAry(des, TGdes, 2 * MAX_ROW, GRID_VOLUME());
#endif

/////// CMatrix-Vector ////////
#if 1
    {
        watcher.reset();
        ComplexAry_MatrixVector(Cdes, CMat, Csrc, GRID_VOLUME());
        // ComplexAry_MatrixVector02(Cdes, CMat, Csrc, GRID_VOLUME());
        // TensorGrid_CMatrixVector(TGdes, TGmat, TGsrc, GRID_VOLUME());
        // TensorGrid_CMatrixVector(TGB, TGMAT, TGA, GRID_VOLUME());
        double timeCMV0 = watcher.use();

        // watcher.reset();
        // ComplexAry_MatrixVector02(Cdes, CMat, Csrc, GRID_VOLUME());
        // double timeCMV02 = watcher.use();
        // printf("Complex Array ACC: timeCMV0 / timeCMV02 = %6.2lf\n", timeCMV0 / timeCMV02);

        /// TGB = TGMAT * TGA;
        watcher.reset();
        TensorGrid_CMatrixVector(TGdes, TGmat, TGsrc, GRID_VOLUME());
        // TensorGrid_CMatrixVector02(TGdes, TGmat, TGsrc, GRID_VOLUME());
        double timeTGMV = watcher.use();

        DataType diffres = diff_Ary_TGAry(des, TGdes, 2 * MAX_ROW, GRID_VOLUME());
        printf("  Gemv: Acc%6.2lf |time CBase %8.2e  TG %8.2e  Tran %8.2e(%4.1lfx) |diff %8.2g | GridSize %ld\n",
               timeCMV0 / timeTGMV, timeCMV0, timeTGMV, timeTrans, timeTrans / timeTGMV, diffres, GRID_VOLUME());
    }
#endif

    delete[] mat;
    delete[] src;
    delete[] des;
    delete[] TGmat;
    delete[] TGsrc;
    delete[] TGdes;

    return 0;
}