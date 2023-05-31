
#include <omp.h>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "tensorgrid_R3.h"
#include "timer.h"
#include "transfer.h"

int main(int argc, char **argv)
{
    timer watcher;
    watcher.reset();

    // printf("sizeof(DataType) = %ld bytes\n", sizeof(DataType));
    ////// Complxe Array //////
    DataType *srcX = new DataType[MAX_COL * GRID_VOLUME()];
    DataType *srcY = new DataType[MAX_COL * GRID_VOLUME()];
    DataType *des = new DataType[MAX_COL * GRID_VOLUME()];
    /// random
    random(srcX, MAX_COL * GRID_VOLUME());
    random(srcY, MAX_COL * GRID_VOLUME());
    // random(des, MAX_COL * GRID_VOLUME());
    memset(des, 0, MAX_COL * GRID_VOLUME() * sizeof(DataType));

    ////// Tensor Grid //////
    DataType *TGsrcX = new DataType[MAX_COL * GRID_VOLUME()];
    DataType *TGsrcY = new DataType[MAX_COL * GRID_VOLUME()];
    DataType *TGdes = new DataType[MAX_COL * GRID_VOLUME()];

    /// repermute data layout
    watcher.reset();
    tranfer2TG(TGsrcX, srcX, MAX_COL, GRID_VOLUME());
    tranfer2TG(TGsrcY, srcY, MAX_COL, GRID_VOLUME());
    tranfer2TG(TGdes, des, MAX_COL, GRID_VOLUME());
    double timeTrans = watcher.use();
    // printf("transfer time: %10.2e\n", timeTrans);

#ifdef debug
    diff_Ary_TGAry(srcX, TGsrcX, MAX_COL, GRID_VOLUME());
#endif

    ////////// Complex Vector dest = XY + TGdes;
    {
        watcher.reset();
        RealArray_RXYpY(des, srcX, srcY, MAX_COL * GRID_VOLUME());
        double timeV = watcher.use();

        watcher.reset();
        TensorGrid_RXYpY(TGdes, TGsrcX, TGsrcY, MAX_COL, GRID_VOLUME());
        double timeTGV = watcher.use();

        DataType diffY = diff_Ary_TGAry(des, TGdes, MAX_COL, GRID_VOLUME());
        printf("  XYpY  Acc:%6.2lf    diff%9.2e  GridSize  %ld\n", timeV / timeTGV, diffY, GRID_VOLUME());

        // watcher.reset();
        // RealArray_RXYpY(srcY, srcX, srcY, MAX_COL * GRID_VOLUME());
        // double timeV2 = watcher.use();

        // watcher.reset();
        // TensorGrid_RXYpY(TGsrcY, TGsrcX, TGsrcY, MAX_COL, GRID_VOLUME());
        // double timeTGV2 = watcher.use();

        // DataType diffY2 = diff_Ary_TGAry(des, TGdes, MAX_COL, GRID_VOLUME());
        // printf("  XYpY  Acc:%6.2lf    diff%9.2e  GridSize  %ld\n", timeV2 / timeTGV2, diffY2, GRID_VOLUME());
    }

    // {
    //     watcher.reset();
    //     TensorGrid_RXYpY(TGdes, TGsrcX, TGsrcY, MAX_COL, GRID_VOLUME());
    //     double timeTGV = watcher.use();

    //     watcher.reset();
    //     TensorGrid_RXYpY(TGsrcY, TGsrcX, TGsrcY, MAX_COL, GRID_VOLUME());
    //     double timeTGV2 = watcher.use();

    //     DataType diffTG = diff_vector_norm2(TGdes, TGsrcY, MAX_COL * GRID_VOLUME());
    //     printf(" CXYpY: res=X*Y+Y v.s. Y = X*Y-Y :  Acc:%6.2lf  diff %8.2e  GridSize  %ld\n", timeTGV / timeTGV2, diffTG,
    //            GRID_VOLUME());
    // }

    delete[] srcX;
    delete[] des;
    delete[] TGsrcX;
    delete[] TGdes;

    return 0;
}