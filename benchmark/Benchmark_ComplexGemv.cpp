/**
 * @file Benchmark_ComplexGemv.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-24
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#define ALIGN_NUM 64

#ifndef LOOPNUM
#define LOOPNUM 100
#endif

#ifndef NRUN
#define NRUN 5
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
int Benchmark_Gemv();

int main(int argc, char const *argv[])
{
    Benchmark_Gemv();
    return 0;
}


int Benchmark_Gemv()
{
    timer watcher;

    // constexpr size_t MAX_GRID = 16 * 16 * 8 * 8;
    constexpr size_t Lmax     = 16;
    constexpr size_t MAX_GRID = Lmax * Lmax * Lmax * Lmax;

    watcher.reset();

#if not defined(CHECK)
    printf("=== The Benchmark for Batched gemv: sizeof(FLOAT)(%ld), M=%d, N=%d,aligned(%d) ===\n",
           sizeof(FLOAT), MROW, NCOL, ALIGN_NUM);
    printf("%8s%20s%12s%12s%12s%12s%12s%12s%12s\n", "Mem(K)", "GridSize =L^4 ", "BaseTime", "cblas",
           "AutoSimd", "Bench_v0", "Bench_v1", "TGBlasv0", "TGBlasv1");
#endif

#if defined(CHECK)
    printf("=== The Accuracy Checking for gemv: sizeof(FLOAT)(%ld), M=%d, N=%d,aligned(%d) ===\n",
           sizeof(FLOAT), MROW, NCOL, ALIGN_NUM);
    printf("%8s%20s%8s%8s%8s%8s%8s%8s%8s%8s\n", "Mem(K)", "GridSize  ~L^4", "dsrc", "dtranf",
           "Cblas", "bchAuto", "bchv0", "bchv1", "TGv0", "TGv1");
#endif

    double timebase, timecblas, timebenchauto, timebchv0, timebchv1, timeTGBlasv0, timeTGBlasv1;
    double ddesbase, ddescblas, ddesbenchauto, ddesbchv0, ddesbchv1, ddesTGBlasv0, ddesTGBlasv1;


    // size_t GridSize = 0;
    for (size_t GridSize = BASEGRID; GridSize <= MAX_GRID; GridSize *= 2) {

        FLOAT *A, *X, *Y;
        A = (FLOAT *) _mm_malloc(2 * MROW * NCOL * GridSize * sizeof(FLOAT), ALIGN_NUM);
        X = (FLOAT *) _mm_malloc(2 * NCOL * GridSize * sizeof(FLOAT), ALIGN_NUM);
        Y = (FLOAT *) _mm_malloc(2 * MROW * GridSize * sizeof(FLOAT), ALIGN_NUM);

        FLOAT *TGA, *TGX, *TGY;
        TGA = (FLOAT *) _mm_malloc(2 * MROW * NCOL * GridSize * sizeof(FLOAT), ALIGN_NUM);
        TGX = (FLOAT *) _mm_malloc(2 * NCOL * GridSize * sizeof(FLOAT), ALIGN_NUM);
        TGY = (FLOAT *) _mm_malloc(2 * MROW * GridSize * sizeof(FLOAT), ALIGN_NUM);

        /// random
        random(A, 2 * MROW * NCOL * GridSize);
        random(X, 2 * NCOL * GridSize);
        tranfer2TG(TGA, A, MROW * NCOL * 2, GridSize);
        tranfer2TG(TGX, X, NCOL * 2, GridSize);

        FLOAT dsrc = diff_Ary_TGAry(X, TGX, 2 * NCOL, GridSize);

        int repeat = NRUN;
        while (repeat--) {

            /////// Base CMatrix-Vector ////////
            watcher.reset();
            for (size_t l = 0; l < LOOPNUM; l++) {
                // AOS_Gemv_batch_v2(MROW, NCOL, A, X, Y, GridSize);
                AOS_Gemv_batch(MROW, NCOL, (std::complex<FLOAT> *) A, (std::complex<FLOAT> *) X,
                               (std::complex<FLOAT> *) Y, GridSize);
                // AOS_Gemv_batch_v1(MROW, NCOL, (std::complex<FLOAT> *) A, (std::complex<FLOAT> *) X,
                //                   (std::complex<FLOAT> *) Y, GridSize);
            }
            timebase = watcher.use();

            tranfer2TG(TGY, Y, MROW * 2, GridSize);
            ddesbase = diff_Ary_TGAry(Y, TGY, 2 * MROW, GridSize);

#if defined(HAVE_BLAS)
            watcher.reset();
            for (size_t l = 0; l < LOOPNUM; l++) {
                ComplexAry_MatrixVector_cblas(MROW, NCOL, (std::complex<FLOAT> *) A,
                                              (std::complex<FLOAT> *) X,
                                              (std::complex<FLOAT> *) TGY, GridSize);
            }
            timecblas = watcher.use();
            ddescblas = diff_vector_norm(Y, TGY, 2 * MROW * GridSize);
#endif

            /////////////// TensorGrid //////////
            watcher.reset();
            for (size_t l = 0; l < LOOPNUM; l++) {
                TensorGrid_CMatrixVector(TGY, TGA, TGX, GridSize);
            }
            timebenchauto = watcher.use();
            ddesbenchauto = diff_Ary_TGAry(Y, TGY, 2 * MROW, GridSize);

#if defined(__AVX512F__)

            watcher.reset();
            for (size_t l = 0; l < LOOPNUM; l++) {
                TensorGrid_CMatrixVector_avx512(TGY, TGA, TGX, MROW, NCOL, GridSize);
            }
            timebchv0 = watcher.use();
            ddesbchv0 = diff_Ary_TGAry(Y, TGY, 2 * MROW, GridSize);

            watcher.reset();
            for (size_t l = 0; l < LOOPNUM; l++) {
                TensorGrid_CMatrixVector_avx512_v1<MROW, NCOL>(TGY, TGA, TGX, GridSize);
            }
            timebchv1 = watcher.use();
            ddesbchv1 = diff_Ary_TGAry(Y, TGY, 2 * MROW, GridSize);

#endif

#if defined(__AVX__) || defined(__AVX2__)
            watcher.reset();
            for (size_t l = 0; l < LOOPNUM; l++) {
                TensorGrid_CMatrixVector_avx256(TGY, TGA, TGX, MROW, NCOL, GridSize);
            }
            timebchv0 = watcher.use();
            ddesbchv0 = diff_Ary_TGAry(Y, TGY, 2 * MROW, GridSize);

            watcher.reset();
            for (size_t l = 0; l < LOOPNUM; l++) {
                TensorGrid_CMatrixVector_avx256_v1<MROW, NCOL>(TGY, TGA, TGX, GridSize);
            }
            timebchv1 = watcher.use();
            ddesbchv1 = diff_Ary_TGAry(Y, TGY, 2 * MROW, GridSize);
#endif

            watcher.reset();
            for (size_t l = 0; l < LOOPNUM; l++) {
                TensorGrid_complex_gemv<MROW, NCOL>(TGA, TGX, TGY, GridSize);
            }
            timeTGBlasv0 = watcher.use();
            ddesTGBlasv0 = diff_Ary_TGAry(Y, TGY, 2 * MROW, GridSize);


            watcher.reset();
            for (size_t l = 0; l < LOOPNUM; l++) {
                // TensorGrid_complex_gemv_v1<MROW, NCOL>(TGA, TGX, TGY, GridSize);
                TensorGrid_complex_gemv_v2<MROW, NCOL>(TGA, TGX, TGY, GridSize);

                // TensorGrid_complex_gemm_v1<MROW, 1, NCOL>(TGA, TGX, TGY, GridSize);
            }
            timeTGBlasv1 = watcher.use();
            ddesTGBlasv1 = diff_Ary_TGAry(Y, TGY, 2 * MROW, GridSize);


            double MemSize = (double) (2 * (MROW * NCOL + MROW + NCOL) * GridSize * sizeof(FLOAT));
            MemSize /= 1024.0;
            double Lreduce = sqrt(sqrt(GridSize));

#if not defined(CHECK)
            printf("%8.1f%12ld%8.2lf%12.2lf%12.3g%12.2f%12.2f%12.2f%12.2f%12.2f\n", MemSize,
                   GridSize, Lreduce, timebase, timebase / timecblas, timebase / timebenchauto,
                   timebase / timebchv0, timebase / timebchv1, timebase / timeTGBlasv0,
                   timebase / timeTGBlasv1);
#endif

#if defined(CHECK)
            printf("%8.1f%12ld%8.2f%8.1g%8.1g%8.1g%8.1g%8.1g%8.1g%8.1g%8.1g\n", MemSize, GridSize,
                   Lreduce, dsrc, ddesbase, ddescblas, ddesbenchauto, ddesbchv0, ddesbchv1,
                   ddesTGBlasv0, ddesTGBlasv1);

#endif
        }

        printf("\n");

        _mm_free(A);
        _mm_free(X);
        _mm_free(Y);
        _mm_free(TGA);
        _mm_free(TGX);
        _mm_free(TGY);
    }

    return 0;
}
