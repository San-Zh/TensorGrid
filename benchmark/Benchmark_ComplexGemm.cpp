/**
 * @file Benchmark_Gemv_MxN.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-20
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

#ifndef MROW
#define MROW 3
#endif

#ifndef NCOL
#define NCOL 3
#endif

#ifndef KIN
#define KIN 3
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
int Benchmark_Gemm();

int main(int argc, char const *argv[])
{
    Benchmark_Gemm();
    return 0;
}


int Benchmark_Gemm()
{
    timer watcher;

    // constexpr size_t MAX_GRID = 16 * 16 * 8 * 8;
    constexpr size_t Lmax     = 16;
    constexpr size_t MAX_GRID = Lmax * Lmax * Lmax * Lmax;

    watcher.reset();
#if not defined(CHECK)
    printf("=== Benchmark for Batched gemm: sizeof(FLOAT)(%ld), M=%d, N=%d,K=%d, aligned(%d) ===\n",
           sizeof(FLOAT), MROW, NCOL, KIN, ALIGN_NUM);
    printf("%8s%20s%12s%12s%12s%12s\n", "Mem(Kb)", "GridSize =L^4 ", "BaseTime", "cblas",
           "TGBlasv0", "TGBlasv1");
#endif

#if defined(CHECK)
    printf("=== The Accuracy Checking for gemv: sizeof(FLOAT)(%ld), M=%d, N=%d,aligned(%d) ===\n",
           sizeof(FLOAT), MROW, NCOL, ALIGN_NUM);
    printf("%8s%20s%8s%8s%8s%8s%8s\n", "Mem(Kb)", "GridSize  ~L^4", "dA", "dtrsfC", "Cblas", "TGv0",
           "TGv1");
#endif

    double t_base, t_cblas, t_TGBlasv0, t_TGBlasv1;
    double dCtrsf, dCcblas, dCTGBlasv0, dCTGBlasv1;

    // size_t GridSize = 0;
    for (size_t GridSize = BASEGRID; GridSize <= MAX_GRID; GridSize *= 2) {

        FLOAT *A, *B, *C;
        A = (FLOAT *) _mm_malloc(2 * MROW * KIN * GridSize * sizeof(FLOAT), ALIGN_NUM);
        B = (FLOAT *) _mm_malloc(2 * KIN * NCOL * GridSize * sizeof(FLOAT), ALIGN_NUM);
        C = (FLOAT *) _mm_malloc(2 * MROW * NCOL * GridSize * sizeof(FLOAT), ALIGN_NUM);

        FLOAT *TGA, *TGB, *TGC;
        TGA = (FLOAT *) _mm_malloc(2 * MROW * KIN * GridSize * sizeof(FLOAT), ALIGN_NUM);
        TGB = (FLOAT *) _mm_malloc(2 * KIN * NCOL * GridSize * sizeof(FLOAT), ALIGN_NUM);
        TGC = (FLOAT *) _mm_malloc(2 * MROW * NCOL * GridSize * sizeof(FLOAT), ALIGN_NUM);

        /// random
        random(A, 2 * MROW * KIN * GridSize);
        random(B, 2 * KIN * NCOL * GridSize);
        tranfer2TG(TGA, A, MROW * KIN * 2, GridSize);
        tranfer2TG(TGB, B, KIN * NCOL * 2, GridSize);

        FLOAT dA, dB;
        dA = diff_Ary_TGAry(A, TGA, 2 * MROW * KIN, GridSize);
        dB = diff_Ary_TGAry(B, TGB, 2 * KIN * NCOL, GridSize);

        int repeat = NRUN;
        while (repeat--) {

            /////// Base CMatrix-Vector ////////
            watcher.reset();
            for (size_t l = 0; l < LOOPNUM; l++) {
                // ComplexAry_MatrixVector_v2<MROW, NCOL>(C, A, B, GridSize);
                AOS_Gemm_batch(MROW, NCOL, KIN, (std::complex<FLOAT> *) A,
                               (std::complex<FLOAT> *) B, (std::complex<FLOAT> *) C, GridSize);
                // ComplexAry_MatrixVector02<MROW, NCOL>((std::complex<FLOAT> *) (C),
                //                                       (std::complex<FLOAT> *) (A),
                //                                       (std::complex<FLOAT> *) (B), GridSize);
            }
            t_base = watcher.use();

            tranfer2TG(TGC, C, 2 * MROW * NCOL, GridSize);
            dCtrsf = diff_Ary_TGAry(C, TGC, 2 * MROW * NCOL, GridSize);

#if defined(HAVE_BLAS)
            watcher.reset();
            for (size_t l = 0; l < LOOPNUM; l++) {
                ComplexAry_MatrixMatrix_cblas(MROW, NCOL, KIN, A, B, TGC, GridSize);
            }
            t_cblas = watcher.use();
            dCcblas = diff_vector_norm(C, TGC, 2 * MROW * NCOL * GridSize);
#endif

            watcher.reset();
            for (size_t l = 0; l < LOOPNUM; l++) {
                TensorGrid_complex_gemm<MROW, NCOL, KIN>(TGA, TGB, TGC, GridSize);
            }
            t_TGBlasv0 = watcher.use();
            dCTGBlasv0 = diff_Ary_TGAry(C, TGC, 2 * MROW * NCOL, GridSize);


            watcher.reset();
            for (size_t l = 0; l < LOOPNUM; l++) {
                TensorGrid_complex_gemm_v1<MROW, NCOL, KIN>(TGA, TGB, TGC, GridSize);
            }
            t_TGBlasv1 = watcher.use();
            dCTGBlasv1 = diff_Ary_TGAry(C, TGC, 2 * MROW * NCOL, GridSize);


            double Lreduce = sqrt(sqrt(GridSize));
            double MemSize =
                (double) (2 * (MROW * KIN + KIN * NCOL + MROW * NCOL) * GridSize * sizeof(FLOAT));
            MemSize /= 1024.0;

#if not defined(CHECK)
            printf("%8.1f%12ld%8.2lf%12.2lf%12.3g%12.2f%12.2f\n", MemSize, GridSize, Lreduce,
                   t_base, t_base / t_cblas, t_base / t_TGBlasv0, t_base / t_TGBlasv1);
#endif

#if defined(CHECK)
            printf("%8.1f%12ld%8.2f%8.1g%8.1g%8.1g%8.1g%8.1g\n", MemSize, GridSize, Lreduce, dA,
                   dCtrsf, dCcblas, dCTGBlasv0, dCTGBlasv1);
#endif
        }

        printf("\n");

        _mm_free(A);
        _mm_free(B);
        _mm_free(C);
        _mm_free(TGA);
        _mm_free(TGB);
        _mm_free(TGC);
    }

    return 0;
}
