
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
    DataType *src = new DataType[2 * MAX_COL * GRID_VOLUME()];
    DataType *mat = new DataType[2 * MAX_ROW * MAX_COL * GRID_VOLUME()];
    DataType *des = new DataType[2 * MAX_COL * GRID_VOLUME()];
    /// random
    random(src, 2 * MAX_COL * GRID_VOLUME());
    random(mat, 2 * MAX_ROW * MAX_COL * GRID_VOLUME());
    random(des, 2 * MAX_COL * GRID_VOLUME());
    // memset(des, 0, 2 * MAX_COL * GRID_VOLUME() * sizeof(DataType));

    ////// Tensor Grid //////
    DataType *TGsrc = new DataType[2 * MAX_COL * GRID_VOLUME()];
    DataType *TGmat = new DataType[2 * MAX_ROW * MAX_COL * GRID_VOLUME()];
    DataType *TGdes = new DataType[2 * MAX_COL * GRID_VOLUME()];
    // memset(TGdes, 0, 2 * MAX_COL * GRID_VOLUME() * sizeof(DataType));

    ////////// Complex Vector dest = XY + TGdes;
    size_t ArySize = 2 * MAX_COL * GRID_VOLUME();
    size_t AryBytes = ArySize * sizeof(DataType);
    {
        watcher.reset();
        AryIO(TGsrc, src, ArySize);
        double timeIO = watcher.use_usec();
        double bandwidth = static_cast<double>(AryBytes) * 1.0e-3 / timeIO;
        printf("  dest <- src :   bandwidth = %.2lf GB/s, time %6.lf us, GridSize  %ld\n", bandwidth, timeIO,
               GRID_VOLUME());
    }

    {
        watcher.reset();
        AryRead(src, ArySize);
        double timeI = watcher.use_usec();
        double bandwidthI = static_cast<double>(AryBytes) * 1.0e-3 / timeI;
        printf("  dest <- src : I-bandwidth = %.2lf GB/s, time %6.lf us, GridSize  %ld\n", bandwidthI, timeI,
               GRID_VOLUME());
    }

    {
        watcher.reset();
        AryWrite(des, ArySize);
        double timeO = watcher.use_usec();
        double bandwidthO = static_cast<double>(AryBytes) * 1.0e-3 / timeO;
        printf("  dest <- src : O-bandwidth = %.2lf GB/s, time %6.lf us, GridSize  %ld\n", bandwidthO, timeO,
               GRID_VOLUME());
    }

    // { /// repermute data layout
    //     watcher.reset();
    //     tranfer2TG(TGsrc, src, 2 * MAX_COL, GRID_VOLUME());
    //     // tranfer2TG(TGdes, des, 2 * MAX_COL, GRID_VOLUME());
    //     double timeTrans = watcher.use_usec();
    //     double bandwidth = static_cast<double>(AryBytes) * 1.0e-3 / timeTrans;
    //     printf("  transfer2TG :   bandwidth = %.2lf GB/s  GridSize  %ld\n", bandwidth, GRID_VOLUME());
    // }

    // { /// repermute data layout
    //     watcher.reset();
    //     tranfer2TG(TGmat, mat, 2 * MAX_ROW * MAX_COL, GRID_VOLUME());
    //     double timeTrans2TG = watcher.use_usec();
    //     double bandwidth2TG = static_cast<double>(AryBytes * MAX_ROW) * 1.0e-3 / timeTrans2TG;

    //     watcher.reset();
    //     tranfer2general(mat, TGmat, 2 * MAX_ROW * MAX_COL, GRID_VOLUME());
    //     double timeTrans2gn = watcher.use_usec();
    //     double bandwidth2gn = static_cast<double>(AryBytes * MAX_ROW) * 1.0e-3 / timeTrans2gn;
    //     printf("  transfe: to-TG bandwidth %.2lf GB/s, to-general bandwidth %.2lf GB/s, Rate %.2lf ,  GridSize  %ld\n",
    //            bandwidth2TG, bandwidth2gn, timeTrans2TG / timeTrans2gn, GRID_VOLUME());
    // }

#ifdef debug
    diff_Ary_TGAry(src, TGsrc, 2 * MAX_COL, GRID_VOLUME());
#endif

    delete[] src;
    delete[] des;
    delete[] TGsrc;
    delete[] TGdes;

    return 0;
}