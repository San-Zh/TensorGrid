/**
 * @file benchmark_su3_mv.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-14
 * 
 * @copyright Copyright (c) 2023
 * 
 */

// #ifndef __AVX512F__
// #define __AVX512F__
// #endif

#define ALIGN_NUM 64

#ifndef LOOPNUM
#define LOOPNUM 10
#endif

#ifndef BASEGRID
#define BASEGRID 64
#endif

#ifndef PRECISION
#define PRECISION DOUBLE
#define FLOAT     double
#else
#define PRECISION SINGLE
#define FLOAT     float
#endif

#include <omp.h>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mm_malloc.h>

#include "timer.h"
#include "transfer.h"
#include "benchmark.h"
#include "TensorGrid_Blas.h"

using namespace std;


#define Debug(e)                                                 \
    do {                                                         \
        printf("debug: " #e " : %s : %d\n", __FILE__, __LINE__); \
    } while (0)



void benchmark_check();

int main(int argc, char **argv)
{

    benchmark_check();
    return 0;
}

void benchmark_check()
{
    timer watcher;

    printf("=== The Accuracy Checking for gemv with sizeof(FLOAT)(%ld), aligned(%d) ===\n",
           sizeof(FLOAT), ALIGN_NUM);

    printf("%8s%20s%8s%8s%12s%12s%12s%12s\n", "Mem(K)", "GridSize ~ L^4", "dsrc", "ddes",
           "AutoSimd", "TensorBlas", "Bench_v2", "v0.0.1");
    constexpr size_t Lmax     = 16;
    constexpr size_t MAX_GRID = Lmax * Lmax * Lmax * Lmax;

    size_t GridSize = BASEGRID;
    // for (GridSize = BASEGRID; GridSize <= MAX_GRID; GridSize += BASEGRID) {
    for (GridSize = BASEGRID; GridSize <= MAX_GRID; GridSize *= 2) {

#ifdef DEBUG
        Debug();
#endif
        watcher.reset();

        FLOAT *src, *mat, *des;
        FLOAT *TGmat, *TGsrc, *TGdes;

        mat = (FLOAT *) _mm_malloc(2 * MROW * NCOL * GridSize * sizeof(FLOAT), ALIGN_NUM);
        src = (FLOAT *) _mm_malloc(2 * NCOL * GridSize * sizeof(FLOAT), ALIGN_NUM);
        des = (FLOAT *) _mm_malloc(2 * MROW * GridSize * sizeof(FLOAT), ALIGN_NUM);


        TGmat = (FLOAT *) _mm_malloc(2 * MROW * NCOL * GridSize * sizeof(FLOAT), ALIGN_NUM);
        TGsrc = (FLOAT *) _mm_malloc(2 * NCOL * GridSize * sizeof(FLOAT), ALIGN_NUM);
        TGdes = (FLOAT *) _mm_malloc(2 * MROW * GridSize * sizeof(FLOAT), ALIGN_NUM);

        double MemSize = (double) (4 * (MROW * NCOL + MROW + NCOL) * GridSize * sizeof(FLOAT));
        MemSize /= 1024.0;
        /// random
        random(mat, 2 * MROW * NCOL * GridSize);
        random(src, 2 * NCOL * GridSize);
        // random(TGmat, 2 * MROW * NCOL * GridSize);
        // random(TGsrc, 2 * NCOL * GridSize);


        // tranfer2general(mat, TGmat, MROW * NCOL * 2, GridSize);
        // tranfer2general(src, mat, MROW * 2, GridSize);


        double timebase, timebenchauto, timeTensorBlas, timebenchv2, timebenchv1;
        double ddesbase, ddesbenchauto, ddesTensorBlas, ddesbenchv2, ddesbenchv1;
        FLOAT  dsrc, dmat, ddes;

        tranfer2TG(TGmat, mat, MROW * NCOL * 2, GridSize);
        tranfer2TG(TGsrc, src, NCOL * 2, GridSize);

        dsrc = diff_Ary_TGAry(src, TGsrc, 2 * NCOL, GridSize);
        dmat = diff_Ary_TGAry(mat, TGmat, 2 * NCOL * MROW, GridSize);

        // memset(TGdes, 0, 2 * MROW * GridSize * sizeof(FLOAT));
        // memset(des, 0, 2 * MROW * GridSize * sizeof(FLOAT));

        /////// Base CMatrix-Vector ////////
        watcher.reset();
        for (size_t l = 0; l < LOOPNUM; l++) {
            // ComplexAry_MatrixVector_v2<MROW, NCOL>(des, mat, src, GridSize);
            // ComplexAry_MatrixVector<MROW, NCOL>((std::complex<FLOAT> *) (des),
            //                                     (std::complex<FLOAT> *) (mat),
            //                                     (std::complex<FLOAT> *) (src), GridSize);
            ComplexAry_MatrixVector02<MROW, NCOL>((std::complex<FLOAT> *) (des),
                                                  (std::complex<FLOAT> *) (mat),
                                                  (std::complex<FLOAT> *) (src), GridSize);
        }

        timebase = watcher.use();
        tranfer2TG(TGdes, des, MROW * 2, GridSize);
        ddes     = diff_Ary_TGAry(des, TGdes, 2 * MROW, GridSize);
        ddesbase = ddes;

        // memset(TGdes, 0, 2 * MROW * GridSize * sizeof(FLOAT));
        // memset(des, 0, 2 * MROW * GridSize * sizeof(FLOAT));


        watcher.reset();
        for (size_t l = 0; l < LOOPNUM; l++) {
            TensorGrid_CMatrixVector(TGdes, TGmat, TGsrc, GridSize);
        }
        timebenchauto = watcher.use();
        ddesbenchauto = diff_Ary_TGAry(des, TGdes, 2 * MROW, GridSize);

        memset(TGdes, 0, 2 * MROW * GridSize * sizeof(FLOAT));

        watcher.reset();
        for (size_t l = 0; l < LOOPNUM; l++) {
            TensorGrid_zgemv_batch<FLOAT, MROW, NCOL>(TGdes, TGmat, TGsrc, GridSize);
        }
        timeTensorBlas = watcher.use();
        ddesTensorBlas = diff_Ary_TGAry(des, TGdes, 2 * MROW, GridSize);
        // ddesTensorBlas = diff_vector_norm2(des, TGdes, 2 * MROW * GridSize);

#if defined(__AVX512F__)

        memset(TGdes, 0, 2 * MROW * GridSize * sizeof(FLOAT));

        watcher.reset();
        for (size_t l = 0; l < LOOPNUM; l++) {
            TensorGrid_CMatrixVector_Batch_avx512<MROW, NCOL>(TGdes, TGmat, TGsrc, GridSize);
        }
        timebenchv2 = watcher.use();
        ddesbenchv2 = diff_Ary_TGAry(des, TGdes, 2 * MROW, GridSize);

        memset(TGdes, 0, 2 * MROW * GridSize * sizeof(FLOAT));

        watcher.reset();
        for (size_t l = 0; l < LOOPNUM; l++) {
            TensorGrid_CMatrixVector_avx512(TGdes, TGmat, TGsrc, MROW, NCOL, GridSize);
        }
        timebenchv1 = watcher.use();
        ddesbenchv1 = diff_Ary_TGAry(des, TGdes, 2 * MROW, GridSize);
        // ddesbenchv1 = diff_vector_norm2(des, TGdes, 2 * MROW * GridSize);

#endif

#if defined(__AVX__) || defined(__AVX2__)

        memset(TGdes, 0, 2 * MROW * GridSize * sizeof(FLOAT));

        watcher.reset();
        for (size_t l = 0; l < LOOPNUM; l++) {
            TensorGrid_CMatrixVector_Batch_avx256<MROW, NCOL>(TGdes, TGmat, TGsrc, GridSize);
        }
        timebenchv2 = watcher.use();
        ddesbenchv2 = diff_Ary_TGAry(des, TGdes, 2 * MROW, GridSize);

        memset(TGdes, 0, 2 * MROW * GridSize * sizeof(FLOAT));


        watcher.reset();
        for (size_t l = 0; l < LOOPNUM; l++) {
            TensorGrid_CMatrixVector_avx256(TGdes, TGmat, TGsrc, MROW, NCOL, GridSize);
        }
        timebenchv1 = watcher.use();
        ddesbenchv1 = diff_Ary_TGAry(des, TGdes, 2 * MROW, GridSize);
        // ddesbenchv1 = diff_vector_norm2(des, TGdes, 2 * MROW * GridSize);

#endif

        double Lreduce = sqrt(sqrt(GridSize));

        printf("%8.1f%12ld%8.2f%8.1g%8.1g%12.2g%12.2g%12.2g%12.2g\n", MemSize, GridSize, Lreduce,
               dsrc, ddesbase, ddesbenchauto, ddesTensorBlas, ddesbenchv2, ddesbenchv1);
        // printf("%16ld%16.2f%16.3g%16.3f%16.3f%16.3f%16.3f\n", GridSize, Lreduce, timebase,
        //        timebase / timebenchauto, timebase / timeTensorBlas, timebase / timebenchv2,
        //        timebase / timebenchv1);
        _mm_free(mat);
        _mm_free(src);
        _mm_free(des);
        _mm_free(TGmat);
        _mm_free(TGsrc);
        _mm_free(TGdes);

#ifdef DEBUG
        Debug(End while);
#endif
    }
}