
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
    DataType *srcX = new DataType[2 * MAX_COL * GRID_VOLUME()];
    DataType *srcY = new DataType[2 * MAX_COL * GRID_VOLUME()];
    DataType *des = new DataType[2 * MAX_COL * GRID_VOLUME()];
    /// random
    random(srcX, 2 * MAX_COL * GRID_VOLUME());
    random(srcY, 2 * MAX_COL * GRID_VOLUME());
    // random(des, 2 * MAX_COL * GRID_VOLUME());
    memset(des, 0, 2 * MAX_COL * GRID_VOLUME() * sizeof(DataType));
    /// Complex Array
    ComplexPtr CsrcX = (ComplexPtr)(srcX);
    ComplexPtr CsrcY = (ComplexPtr)(srcY);
    ComplexPtr Cdes = (ComplexPtr)(des);

    ////// Tensor Grid //////
    DataType *TGsrcX = new DataType[2 * MAX_COL * GRID_VOLUME()];
    DataType *TGsrcY = new DataType[2 * MAX_COL * GRID_VOLUME()];
    DataType *TGdes = new DataType[2 * MAX_COL * GRID_VOLUME()];

    /// repermute data layout
    watcher.reset();
    tranfer2TG(TGsrcX, srcX, 2 * MAX_COL, GRID_VOLUME());
    tranfer2TG(TGsrcY, srcY, 2 * MAX_COL, GRID_VOLUME());
    tranfer2TG(TGdes, des, 2 * MAX_COL, GRID_VOLUME());
    double timeTrans = watcher.use();
    // printf("transfer time: %10.2e\n", timeTrans);

#ifdef debug
    diff_Ary_TGAry(srcX, TGsrcX, 2 * MAX_COL, GRID_VOLUME());
#endif

    ////////// Complex Vector dest = XY + TGdes;
    {
        watcher.reset();
        ComplexAry_CXYpY(Cdes, CsrcX, CsrcY, MAX_COL * GRID_VOLUME());
        double timeCV = watcher.use();

        watcher.reset();
        TensorGrid_CXYpY(TGdes, TGsrcX, TGsrcY, MAX_COL, GRID_VOLUME());
        double timeTGV = watcher.use();

        DataType diffY = diff_Ary_TGAry(des, TGdes, 2 * MAX_COL, GRID_VOLUME());
        printf("  XYpY  Acc:%6.2lf    diff%9.2e  GridSize  %ld\n", timeCV / timeTGV, diffY, GRID_VOLUME());

        // watcher.reset();
        // ComplexAry_CXYpY(CsrcY, CsrcX, CsrcY, MAX_COL * GRID_VOLUME());
        // double timeCV2 = watcher.use();

        // watcher.reset();
        // TensorGrid_CXYpY(TGsrcY, TGsrcX, TGsrcY, MAX_COL, GRID_VOLUME());
        // double timeTGV2 = watcher.use();

        // DataType diffY2 = diff_Ary_TGAry(des, TGdes, 2 * MAX_COL, GRID_VOLUME());
        // printf("  XYpY  Acc:%6.2lf    diff%9.2e  GridSize  %ld\n", timeCV2 / timeTGV2, diffY2, GRID_VOLUME());
    }

    {
        // watcher.reset();
        // ComplexAry_CXTY(Cdes, CsrcX, CsrcY, MAX_COL, GRID_VOLUME());
        // double timeCVT = watcher.use();

        // watcher.reset();
        // TensorGrid_CXTY(TGdes, TGsrcX, TGsrcY, MAX_COL, GRID_VOLUME());
        // double timeTGCVT = watcher.use();

        // DataType diffVTV = diff_Ary_TGAry(des, TGdes, 2 * MAX_COL, GRID_VOLUME());
        // printf("  CXTYpY  Acc:%6.2lf    diff%9.2e  GridSize  %ld\n", timeCVT / timeTGCVT, diffVTV, GRID_VOLUME());
    }

    delete[] srcX;
    delete[] des;
    delete[] TGsrcX;
    delete[] TGdes;

    return 0;
}