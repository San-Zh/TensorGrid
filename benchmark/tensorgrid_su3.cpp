
#include <omp.h>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "timer.h"
#include "transfer.h"
#include "tensorgrid_su3.h"
// #include "tensorgrid_su3_avx.h"
#include "TensorGrid_Blas.h"
#include "TensorGrid_Blas_avx.h"
#include "TensorGrid_Blas_avx512.h"
#include <mm_malloc.h>

#define ALIGN_NUM 64

#ifndef LOOPNUM
#define LOOPNUM 1
#endif

#define FORLOOPN(e) \
    for (int _lp = 0; _lp < LOOPNUM; _lp++) { e; }

int main(int argc, char **argv)
{
    // #pragma massage(SIMD_OPT)

    timer watcher;
    watcher.reset();
    ////// Complxe Array //////
    DataType *mat = (DataType *) _mm_malloc(
        2 * MAX_ROW * MAX_COL * GRID_VOLUME() * sizeof(DataType), ALIGN_NUM);
    DataType *src =
        (DataType *) _mm_malloc(2 * MAX_COL * GRID_VOLUME() * sizeof(DataType), ALIGN_NUM);
    DataType *des =
        (DataType *) _mm_malloc(2 * MAX_ROW * GRID_VOLUME() * sizeof(DataType), ALIGN_NUM);
    /// random
    random(src, 2 * MAX_COL * GRID_VOLUME());
    random(mat, 2 * MAX_ROW * MAX_COL * GRID_VOLUME());
    memset(des, 0, 2 * MAX_ROW * GRID_VOLUME() * sizeof(DataType));
    /// Complex Array
    ComplexPtr CMat = (ComplexPtr) (mat);
    ComplexPtr Csrc = (ComplexPtr) (src);
    ComplexPtr Cdes = (ComplexPtr) (des);

    ////// Tensor Grid //////
    DataType *TGmat = (DataType *) _mm_malloc(
        2 * MAX_ROW * MAX_COL * GRID_VOLUME() * sizeof(DataType), ALIGN_NUM);
    DataType *TGsrc =
        (DataType *) _mm_malloc(2 * MAX_COL * GRID_VOLUME() * sizeof(DataType), ALIGN_NUM);
    DataType *TGdes =
        (DataType *) _mm_malloc(2 * MAX_ROW * GRID_VOLUME() * sizeof(DataType), ALIGN_NUM);

    /// repermute data layout
    watcher.reset();
    tranfer2TG(TGmat, mat, 2 * MAX_ROW * MAX_COL, GRID_VOLUME());
    tranfer2TG(TGsrc, src, 2 * MAX_COL, GRID_VOLUME());
    tranfer2TG(TGdes, des, 2 * MAX_ROW, GRID_VOLUME());
    double timeTrans = watcher.use();

    // memset(TGdes, 0, 2 * MAX_ROW * GRID_VOLUME() * sizeof(DataType));
#ifdef debug
    std::cout << " src: AoS  -  SoA " << std::endl;
    diff_Ary_TGAry(src, TGsrc, 2 * MAX_COL, GRID_VOLUME());
    std::cout << " mat: AoS  -  SoA " << std::endl;
    diff_Ary_TGAry(mat, TGmat, 2 * MAX_ROW * MAX_COL, GRID_VOLUME());
    // diff_Ary_TGAry(des, TGdes, 2 * MAX_ROW, GRID_VOLUME());
#endif

    /////// Base CMatrix-Vector ////////
    watcher.reset();
    {
        for (size_t l = 0; l < LOOPNUM; l++) {
            ComplexAry_MatrixVector(Cdes, CMat, Csrc, GRID_VOLUME());
            // ComplexAry_MatrixVector02(Cdes, CMat, Csrc, GRID_VOLUME());
            // ComplexAry_MatrixVector_v2(des, mat, src, GRID_VOLUME());

#if defined(__AVX512F__)

            // TensorGrid_CMatrixVector_avx512(TGdes, TGmat, TGsrc, GRID_VOLUME());
            // TensorGrid_zgemv_batch<DataType, MAX_ROW, MAX_COL>(des, TGmat, TGsrc, GRID_VOLUME());
            //     TensorGrid_CMatrixVector_Batch_avx512<3, 3>(des, TGmat, TGsrc, GRID_VOLUME());
            //     TensorGrid_CMatrixVector_avx512(des, TGmat, TGsrc, GRID_VOLUME());
            //     TensorGrid_CMatrixVector_avx512_expand(des, TGmat, TGsrc, GRID_VOLUME());
#endif
        }
    }
    double timeCMV0 = watcher.use();

    watcher.reset();
    {
        for (size_t l = 0; l < LOOPNUM; l++) {
#if defined(__AVX512F__)
            TensorGrid_zgemv_batch<DataType, MAX_ROW, MAX_COL>(TGdes, TGmat, TGsrc, GRID_VOLUME());
            // TensorGrid_CMatrixVector_Batch_avx512<MAX_ROW, MAX_COL>(TGdes, TGmat, TGsrc,
            //                                                         GRID_VOLUME());
            // TensorGrid_CMatrixVector_avx512(TGdes, TGmat, TGsrc, GRID_VOLUME());
            // TensorGrid_CMatrixVector_avx512_expand(TGdes, TGmat, TGsrc, GRID_VOLUME());
#endif

#ifdef __AVX__

            TensorGrid_CMatrixVector_Batch_avx256<MAX_ROW, MAX_COL>(TGdes, TGmat, TGsrc,
                                                                    GRID_VOLUME());
            // TensorGrid_CMatrixVector_Batch_avx256_expand<MAX_ROW, MAX_COL>(TGdes, TGmat, TGsrc,
            //                                                          GRID_VOLUME());
#endif

            // TensorGrid_CMatrixVector(TGdes, TGmat, TGsrc, GRID_VOLUME());
            // ComplexAry_MatrixVector02(Cdes, CMat, Csrc, GRID_VOLUME());
            // ComplexAry_MatrixVector_v2(des, mat, src, GRID_VOLUME());
        }
    }
    double timeTGMV = watcher.use();

#ifdef debug
    std::cout << " des: AoS  -  SoA " << std::endl;
#endif
    DataType diffres = diff_Ary_TGAry(des, TGdes, 2 * MAX_ROW, GRID_VOLUME());
    // auto diffres = diff_vector_norm2(des, TGdes, 2 * MAX_ROW * GRID_VOLUME());
    printf("  Gemv: Acc%6.2lf |time CBase %8.2e TG %8.2e Tran %8.2e (%4.1lf x)|diff %8.2g | "
           "GridSize %ld\n",
           timeCMV0 / timeTGMV, timeCMV0, timeTGMV, timeTrans, timeTrans / timeTGMV, diffres,
           GRID_VOLUME());

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