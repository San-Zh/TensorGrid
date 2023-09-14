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
#define LOOPNUM 100
#endif

#ifndef SIZE
#define SIZE 1024
#endif

// #ifndef PRECISION
// #define PRECISION DOUBLE
// #define FLOAT     double
// #else
// #define PRECISION SINGLE
// #define FLOAT     float
// #endif
#define FLOAT double


#include <omp.h>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <mm_malloc.h>

#include "timer.h"
#include "transfer.h"
#include "benchmark.h"
#include "TensorGrid_Blas.h"

using namespace std;

#define FORLOOPN(e) \
    for (int _lp = 0; _lp < LOOPNUM; _lp++) { e; }

int main(int argc, char **argv)
{
    timer watcher;
    watcher.reset();

    size_t GridSize = SIZE;

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
    tranfer2TG(mat, TGmat, MROW * NCOL * 2, GridSize);
    tranfer2TG(src, mat, MROW * 2, GridSize);

    memset(TGdes, 0, 2 * MROW * GridSize * sizeof(FLOAT));
    memset(des, 0, 2 * MROW * GridSize * sizeof(FLOAT));

    // memset(des, 0, 2 * MROW * GridSize * sizeof(FLOAT));
    // memset(TGdes, 0, 2 * MROW * GridSize * sizeof(FLOAT));

#if defined(DEBUG)
    FLOAT dsrc = diff_Ary_TGAry(src, TGsrc, 2 * NCOL, GridSize);
    std::cout << " src: AoS  -  SoA : |src  -TGsrc| = " << dsrc << std::endl;
    FLOAT dmat = diff_Ary_TGAry(mat, TGmat, 2 * NCOL * MROW, GridSize);
    std::cout << " mat: AoS  -  SoA : |mat  -TGmat| = " << dsrc << std::endl;
#endif

    double ddes = 123.456;

    for (size_t ng = 0; ng < 10; ng++) {

        /////// Base CMatrix-Vector ////////
        watcher.reset();
        for (size_t l = 0; l < LOOPNUM; l++) {
            TensorGrid_CMatrixVector(des, mat, src, GridSize);
            // ComplexAry_MatrixVector02(Cdes, CMat, Csrc, GridSize);
            // ComplexAry_MatrixVector_v2(des, mat, src, GridSize);
        }
        double timebase = watcher.use();

        watcher.reset();
        for (size_t l = 0; l < LOOPNUM; l++) {
            TensorGrid_CMatrixVector(TGdes, TGmat, TGsrc, GridSize);
        }
        double timebenchauto = watcher.use();

        ddes = diff_Ary_TGAry(des, TGdes, 2 * MROW, GridSize);
        cout << " timebase " << timebase << ", ACC " << timebase / timebenchauto
             << ", |des-TGdes| = " << ddes << endl;


        watcher.reset();
        for (size_t l = 0; l < LOOPNUM; l++) {
            TensorGrid_zgemv_batch<FLOAT, MROW, NCOL>(TGdes, TGmat, TGsrc, GridSize);
        }
        double timeTensorBlas = watcher.use();

        ddes = diff_Ary_TGAry(des, TGdes, 2 * MROW, GridSize);
        cout << " timebase " << timebase << ", ACC " << timebase / timeTensorBlas
             << ", |des-TGdes| = " << ddes << endl;


#if defined(__AVX512F__)

        watcher.reset();
        for (size_t l = 0; l < LOOPNUM; l++) {
            TensorGrid_CMatrixVector_Batch_avx512<MROW, NCOL>(TGdes, TGmat, TGsrc, GridSize);
        }
        double timebenchv2 = watcher.use();

        ddes = diff_Ary_TGAry(des, TGdes, 2 * MROW, GridSize);
        cout << " timebase " << timebase << ", ACC " << timebase / timebenchv2
             << ", |des-TGdes| = " << ddes << endl;


        watcher.reset();
        for (size_t l = 0; l < LOOPNUM; l++) {
            TensorGrid_CMatrixVector_avx512(TGdes, TGmat, TGsrc, GridSize);
        }
        double timebenchv1 = watcher.use();

        ddes = diff_Ary_TGAry(des, TGdes, 2 * MROW, GridSize);
        cout << " timebase " << timebase << ", ACC " << timebase / timebenchv1
             << ", |des-TGdes| = " << ddes << endl;

#endif

#if defined(__AVX__)
        // for (size_t l = 0; l < LOOPNUM; l++) {
        //     TensorGrid_CMatrixVector_Batch_avx256<MROW, NCOL>(TGdes, TGmat, TGsrc, GridSize);
        // }
        // double timebenchv1 = watcher.use();
        // for (size_t l = 0; l < LOOPNUM; l++) {}
        // double timebenchv2 = watcher.use();
#endif



        // printf("  Gemv: Acc%6.2lf |time CBase %8.2e TG %8.2e |diff %8.2g | GridSize %ld\n",
        //        timebase / timeBench, timebase, timeBench, diffres, GridSize);
    }

    free(des);
    free(TGmat);
    free(TGsrc);
    free(TGdes);

    return 0;
}