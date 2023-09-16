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
#define LOOPNUM 5
#endif

#ifndef BASEGRID
#define BASEGRID 64
#endif

#ifndef PRECISION
#define FLOAT double
#else
#define PRECISION SINGLE
#define FLOAT     float
#endif
// #define FLOAT double


#define Debug(e)                                                 \
    do {                                                         \
        printf("debug: " #e " : %s : %d\n", __FILE__, __LINE__); \
    } while (0)


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

int main(int argc, char **argv)
{
    constexpr size_t Lmax     = 16;
    constexpr size_t MAX_GRID = Lmax * Lmax * Lmax * Lmax;

    timer watcher;
    watcher.reset();

    printf(" The Benchmark for Benched gemv with sizeof(FLOAT) = %ld\n", sizeof(FLOAT));
    printf("%8s%20s%12s%12s%12s%12s%12s\n", "Mem(K)", "GridSize = L^4", "BaseTime", "AutoSimD",
           "TensorBlas", "Bench_v2", "Bench_v1");

    double timebase, timebenchauto, timeTensorBlas, timebenchv2, timebenchv1;
    double ddesbase, ddesbenchauto, ddesTensorBlas, ddesbenchv2, ddesbenchv1;
    FLOAT  dsrc, dmat, ddes;

    // size_t GridSize = 0;
    size_t GridSize = BASEGRID;

    while (GridSize <= MAX_GRID) {
        ////// Tensor Grid //////
        FLOAT *TGmat, *TGsrc, *TGdes, *src, *mat, *des;
        mat   = (FLOAT *) aligned_alloc(ALIGN_NUM, 2 * MROW * NCOL * GridSize * sizeof(FLOAT));
        src   = (FLOAT *) aligned_alloc(ALIGN_NUM, 2 * NCOL * GridSize * sizeof(FLOAT));
        des   = (FLOAT *) aligned_alloc(ALIGN_NUM, 2 * MROW * GridSize * sizeof(FLOAT));
        TGmat = (FLOAT *) aligned_alloc(ALIGN_NUM, 2 * MROW * NCOL * GridSize * sizeof(FLOAT));
        TGsrc = (FLOAT *) aligned_alloc(ALIGN_NUM, 2 * NCOL * GridSize * sizeof(FLOAT));
        TGdes = (FLOAT *) aligned_alloc(ALIGN_NUM, 2 * MROW * GridSize * sizeof(FLOAT));

        /// random
        // random(TGmat, 2 * MROW * NCOL * GridSize);
        // random(TGsrc, 2 * NCOL * GridSize);
        // tranfer2general(mat, TGmat, MROW * NCOL * 2, GridSize);
        // tranfer2general(src, mat, MROW * 2, GridSize);

        random(mat, 2 * MROW * NCOL * GridSize);
        random(src, 2 * NCOL * GridSize);
        tranfer2TG(TGmat, mat, MROW * NCOL * 2, GridSize);
        tranfer2TG(TGsrc, src, MROW * 2, GridSize);
        // memset(TGdes, 0, 2 * MROW * GridSize * sizeof(FLOAT));
        // memset(des, 0, 2 * MROW * GridSize * sizeof(FLOAT));

        double Lreduce = sqrt(sqrt(GridSize));

        double MemSize = (double) (4 * (MROW * NCOL + MROW + NCOL) * GridSize * sizeof(FLOAT));
        MemSize /= 1024.0;


#if defined(DEBUG)
        dsrc = diff_Ary_TGAry(src, TGsrc, 2 * NCOL, GridSize);
        dmat = diff_Ary_TGAry(mat, TGmat, 2 * NCOL * MROW, GridSize);
#endif

        int repeat = 5;
        while (repeat--) {
            /////// Base CMatrix-Vector ////////
            watcher.reset();
            for (size_t l = 0; l < LOOPNUM; l++) {
                // TensorGrid_CMatrixVector(des, mat, src, GridSize);
                // ComplexAry_MatrixVector_v2<MROW, NCOL>(des, mat, src, GridSize);
                ComplexAry_MatrixVector02<MROW, NCOL>((std::complex<FLOAT> *) (des),
                                                      (std::complex<FLOAT> *) (mat),
                                                      (std::complex<FLOAT> *) (src), GridSize);
            }
            timebase = watcher.use();

            tranfer2TG(TGdes, des, MROW * 2, GridSize);
            ddes     = diff_Ary_TGAry(des, TGdes, 2 * MROW, GridSize);
            ddesbase = ddes;


            watcher.reset();
            for (size_t l = 0; l < LOOPNUM; l++) {
                TensorGrid_CMatrixVector(TGdes, TGmat, TGsrc, GridSize);
            }
            timebenchauto = watcher.use();
            ddesbenchauto = diff_Ary_TGAry(des, TGdes, 2 * MROW, GridSize);
            // cout << "\n ACC " << timebase / timebenchauto << " timebase " << timebase
            //      << ", |des-TGdes| = " << ddes << endl;


            watcher.reset();
            for (size_t l = 0; l < LOOPNUM; l++) {
                TensorGrid_zgemv_batch<FLOAT, MROW, NCOL>(TGdes, TGmat, TGsrc, GridSize);
            }
            timeTensorBlas = watcher.use();
            ddesTensorBlas = diff_Ary_TGAry(des, TGdes, 2 * MROW, GridSize);


#if defined(__AVX512F__)
            watcher.reset();
            for (size_t l = 0; l < LOOPNUM; l++) {
                TensorGrid_CMatrixVector_Batch_avx512<MROW, NCOL>(TGdes, TGmat, TGsrc, GridSize);
            }
            timebenchv2 = watcher.use();
            ddesbenchv2 = diff_Ary_TGAry(des, TGdes, 2 * MROW, GridSize);

            watcher.reset();
            for (size_t l = 0; l < LOOPNUM; l++) {
                TensorGrid_CMatrixVector_avx512(TGdes, TGmat, TGsrc, MROW, NCOL, GridSize);
            }
            timebenchv1 = watcher.use();
            ddesbenchv1 = diff_Ary_TGAry(des, TGdes, 2 * MROW, GridSize);
            // ddesbenchv1 = diff_vector_norm2(des, TGdes, 2 * MROW * GridSize);

#endif

#if defined(__AVX__)
            watcher.reset();
            for (size_t l = 0; l < LOOPNUM; l++) {
                TensorGrid_CMatrixVector_Batch_avx256<MROW, NCOL>(TGdes, TGmat, TGsrc, GridSize);
            }
            timebenchv2 = watcher.use();
            ddesbenchv2 = diff_Ary_TGAry(des, TGdes, 2 * MROW, GridSize);

            watcher.reset();
            for (size_t l = 0; l < LOOPNUM; l++) {
                TensorGrid_CMatrixVector_avx256(TGdes, TGmat, TGsrc, MROW, NCOL, GridSize);
            }
            timebenchv1 = watcher.use();
            ddesbenchv1 = diff_Ary_TGAry(des, TGdes, 2 * MROW, GridSize);
#endif



#if defined(DEBUG)
            printf("%16ld%16.2f%16.3g%16.3g%16.3g%16.3g%16.3g\n", GridSize, Lreduce, ddesbase,
                   ddesbenchauto, ddesTensorBlas, ddesbenchv1, ddesbenchv2);
#endif
            printf("%8.1f%12ld%8.2lf%12.3g%12.2f%12.2f%12.2f%12.2f\n", MemSize, GridSize, Lreduce,
                   timebase, timebase / timebenchauto, timebase / timeTensorBlas,
                   timebase / timebenchv2, timebase / timebenchv1);
        }

        printf("\n");

        free(mat);
        free(src);
        free(des);
        free(TGmat);
        free(TGsrc);
        free(TGdes);

        GridSize *= 2;
    }

    return 0;
}