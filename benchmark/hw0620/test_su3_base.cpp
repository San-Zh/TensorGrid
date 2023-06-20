
#include <omp.h>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "complex_base.h"
#include "timer.h"

#define ALIGN_NUM 64

int main(int argc, char **argv)
{
    timer watcher;
    watcher.reset();
    ////// Complxe Array //////
    DataType *mat = (DataType *) malloc(2 * MAX_ROW * MAX_COL * GRID_VOLUME() * sizeof(DataType));
    DataType *src = (DataType *) malloc(2 * MAX_COL * GRID_VOLUME() * sizeof(DataType));
    DataType *des = (DataType *) malloc(2 * MAX_ROW * GRID_VOLUME() * sizeof(DataType));
    /// random
    random(src, 2 * MAX_COL * GRID_VOLUME());
    random(mat, 2 * MAX_ROW * MAX_COL * GRID_VOLUME());
    memset(des, 0, 2 * MAX_ROW * GRID_VOLUME() * sizeof(DataType));
    /// Complex Array
    ComplexPtr CMat = (ComplexPtr)(mat);
    ComplexPtr Csrc = (ComplexPtr)(src);
    ComplexPtr Cdes = (ComplexPtr)(des);

    /////// CMatrix-Vector ////////
    watcher.reset();
    ComplexAry_MatrixVector(Cdes, CMat, Csrc, GRID_VOLUME());
    double timeCMV0 = watcher.use();
    printf("  Gemv: time CBase %8.2e secs. | GridSize %ld\n", timeCMV0, GRID_VOLUME());

    free(mat);
    free(src);
    free(des);

    return 0;
}