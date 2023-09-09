
#include <omp.h>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "tensorgrid_su3.h"
#include "tensorgrid_su3_avx.h"
#include "timer.h"
#include "transfer.h"

#define ALIGN_NUM 64

#ifndef LOOPNUM
#define LOOPNUM 1000
#endif
#define FORLOOPN(e)                                                                                                    \
    {                                                                                                                  \
        for (int _lp = 0; _lp < LOOPNUM; _lp++) {                                                                      \
            e;                                                                                                         \
        }                                                                                                              \
    }

int main(int argc, char **argv)
{
    timer watcher;
    watcher.reset();
    ////// Complxe Array //////
    DataType *mat = (DataType *) _mm_malloc(2 * MAX_ROW * MAX_COL * GRID_VOLUME() * sizeof(DataType), ALIGN_NUM);
    DataType *src = (DataType *) _mm_malloc(2 * MAX_COL * GRID_VOLUME() * sizeof(DataType), ALIGN_NUM);
    DataType *des = (DataType *) _mm_malloc(2 * MAX_ROW * GRID_VOLUME() * sizeof(DataType), ALIGN_NUM);
    /// random
    random(src, 2 * MAX_COL * GRID_VOLUME());
    random(mat, 2 * MAX_ROW * MAX_COL * GRID_VOLUME());
    memset(des, 0, 2 * MAX_ROW * GRID_VOLUME() * sizeof(DataType));
    /// Complex Array
    ComplexPtr CMat = (ComplexPtr)(mat);
    ComplexPtr Csrc = (ComplexPtr)(src);
    ComplexPtr Cdes = (ComplexPtr)(des);

    ////// Tensor Grid //////
    DataType *TGmat = (DataType *) _mm_malloc(2 * MAX_ROW * MAX_COL * GRID_VOLUME() * sizeof(DataType), ALIGN_NUM);
    DataType *TGsrc = (DataType *) _mm_malloc(2 * MAX_COL * GRID_VOLUME() * sizeof(DataType), ALIGN_NUM);
    DataType *TGdes = (DataType *) _mm_malloc(2 * MAX_ROW * GRID_VOLUME() * sizeof(DataType), ALIGN_NUM);

    /// repermute data layout
    watcher.reset();
    tranfer2TG(TGmat, mat, 2 * MAX_ROW * MAX_COL, GRID_VOLUME());
    tranfer2TG(TGsrc, src, 2 * MAX_COL, GRID_VOLUME());
    tranfer2TG(TGdes, des, 2 * MAX_ROW, GRID_VOLUME());
    double timeTrans = watcher.use();

    // memset(TGdes, 0, 2 * MAX_ROW * GRID_VOLUME() * sizeof(DataType));
#ifdef debug
    diff_Ary_TGAry(src, TGsrc, 2 * MAX_COL, GRID_VOLUME());
    diff_Ary_TGAry(mat, TGmat, 2 * MAX_ROW * MAX_COL, GRID_VOLUME());
    // diff_Ary_TGAry(des, TGdes, 2 * MAX_ROW, GRID_VOLUME());
#endif

    /////// CMatrix-Vector ////////
    {
        watcher.reset();
        // FORLOOPN(TensorGrid_CMatrixVector_avx512(TGdes, TGmat, TGsrc, GRID_VOLUME()));
        FORLOOPN(ComplexAry_MatrixVector(Cdes, CMat, Csrc, GRID_VOLUME()));
        // ComplexAry_MatrixVector02(Cdes, CMat, Csrc, GRID_VOLUME());
        // ComplexAry_MatrixVector_v2(des, mat, src, GRID_VOLUME());
        double timeCMV0 = watcher.use();

        // watcher.reset();
        // ComplexAry_MatrixVector02(Cdes, CMat, Csrc, GRID_VOLUME());
        // double timeCMV02 = watcher.use();
        // printf("Complex Array ACC: timeCMV0 / timeCMV02 = %6.2lf\n", timeCMV0 / timeCMV02);

        /// TGB = TGMAT * TGA;
        watcher.reset();
#if defined __AVX512F__
        // FORLOOPN(TensorGrid_CMatrixVector_avx512(TGdes, TGmat, TGsrc, GRID_VOLUME()));
        FORLOOPN(TensorGrid_CMatrixVector_avx512_expand(TGdes, TGmat, TGsrc, GRID_VOLUME()));
#elif defined __AVX__
        FORLOOPN(TensorGrid_CMatrixVector_avx256(TGdes, TGmat, TGsrc, GRID_VOLUME()));
        // TensorGrid_CMatrixVector_avx256_expand(TGdes, TGmat, TGsrc, GRID_VOLUME());
        // std::cout << "defined __AVX__" << std::endl;
#else
        FORLOOPN(TensorGrid_CMatrixVector(TGdes, TGmat, TGsrc, GRID_VOLUME()));
        // ComplexAry_MatrixVector02(Cdes, CMat, Csrc, GRID_VOLUME());
        // ComplexAry_MatrixVector_v2(des, mat, src, GRID_VOLUME());
#endif
        double timeTGMV = watcher.use();

        DataType diffres = diff_Ary_TGAry(des, TGdes, 2 * MAX_ROW, GRID_VOLUME());
        printf("  Gemv: Acc%6.2lf |time CBase %8.2e TG %8.2e Tran %8.2e (%4.1lf x)|diff %8.2g | GridSize %ld\n",
               timeCMV0 / timeTGMV, timeCMV0, timeTGMV, timeTrans, timeTrans / timeTGMV, diffres, GRID_VOLUME());
    }

    // delete[] mat;
    // delete[] src;
    // delete[] des;
    // delete[] TGmat;
    // delete[] TGsrc;
    // delete[] TGdes;

    free(mat);
    free(src);
    free(des);
    free(TGmat);
    free(TGsrc);
    free(TGdes);

    return 0;
}