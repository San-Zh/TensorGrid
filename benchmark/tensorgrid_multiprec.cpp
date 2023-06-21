
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

int main(int argc, char **argv)
{
    timer watcher;
    watcher.reset();

    using DataTypeD = double;
    using DataTypeF = float;

    ////// Tensor Grid Malloc //////
    DataTypeD *TGmatD = (DataTypeD *) _mm_malloc(2 * MAX_ROW * MAX_COL * GRID_VOLUME() * sizeof(DataTypeD), ALIGN_NUM);
    DataTypeD *TGsrcD = (DataTypeD *) _mm_malloc(2 * MAX_COL * GRID_VOLUME() * sizeof(DataTypeD), ALIGN_NUM);
    DataTypeD *TGdesD = (DataTypeD *) _mm_malloc(2 * MAX_ROW * GRID_VOLUME() * sizeof(DataTypeD), ALIGN_NUM);
    random(TGmatD, 2 * MAX_ROW * MAX_COL * GRID_VOLUME());
    random(TGsrcD, 2 * MAX_COL * GRID_VOLUME());

    DataTypeF *TGmatF = (DataTypeF *) _mm_malloc(2 * MAX_ROW * MAX_COL * GRID_VOLUME() * sizeof(DataTypeF), ALIGN_NUM);
    DataTypeF *TGsrcF = (DataTypeF *) _mm_malloc(2 * MAX_COL * GRID_VOLUME() * sizeof(DataTypeF), ALIGN_NUM);
    DataTypeF *TGdesF = (DataTypeF *) _mm_malloc(2 * MAX_ROW * GRID_VOLUME() * sizeof(DataTypeF), ALIGN_NUM);
    xeqy(TGmatF, TGmatD, 2 * MAX_ROW * MAX_COL * GRID_VOLUME());
    xeqy(TGsrcF, TGsrcD, 2 * MAX_COL * GRID_VOLUME());

    watcher.reset();

    memset(TGdesD, 0, 2 * MAX_ROW * GRID_VOLUME() * sizeof(DataTypeD));
    memset(TGdesF, 0, 2 * MAX_ROW * GRID_VOLUME() * sizeof(DataTypeF));

    /////// CMatrix-Vector ////////
    // {
    //     watcher.reset();
    //     ComplexAry_MatrixVector((std::complex<DataTypeD> *) TGdesD, (std::complex<DataTypeD> *) TGmatD,
    //                             (std::complex<DataTypeD> *) TGsrcD, GRID_VOLUME());
    //     double timeCMD0 = watcher.use();
    //     watcher.reset();
    //     ComplexAry_MatrixVector((std::complex<DataTypeF> *) TGdesF, (std::complex<DataTypeF> *) TGmatF,
    //                             (std::complex<DataTypeF> *) TGsrcF, GRID_VOLUME());
    //     // ComplexAry_MatrixVector(TGdesD, TGmatD, TGsrcD, GRID_VOLUME());
    //     double timeCMF0 = watcher.use();

    //     auto diffres = diff_vector_norm2(TGdesD, TGdesF, 2 * MAX_ROW * GRID_VOLUME());
    //     printf("  Gemv(D/F): Acc%6.2lf |time D %8.2e F %8.2e | diff %8.2g | GridSize %ld (L%5.1lf)\n", timeCMD0 / timeCMF0,
    //            timeCMD0, timeCMF0, diffres, GRID_VOLUME(),sqrt(sqrt(GRID_VOLUME())));
    // }

    // {
    //     watcher.reset();
    //     ComplexAry_MatrixVector(TGdesD, TGmatD, TGsrcD, GRID_VOLUME());
    //     double timeCMD0 = watcher.use();
    //     watcher.reset();
    //     ComplexAry_MatrixVector(TGdesF, TGmatF, TGsrcF, GRID_VOLUME());
    //     double timeCMF0 = watcher.use();
    //     auto diffres = diff_vector_norm2(TGdesD, TGdesF, 2 * MAX_ROW * GRID_VOLUME());
    //     printf("  Gemv(D/F): Acc%6.2lf |time D %8.2e F %8.2e | diff %8.2g | GridSize %ld (L%5.1lf)\n",
    //            timeCMD0 / timeCMF0, timeCMD0, timeCMF0, diffres, GRID_VOLUME(), sqrt(sqrt(GRID_VOLUME())));
    // }

#if 1
    {
        watcher.reset();
#ifdef __AVX512F__
        TensorGrid_CMatrixVector_avx512(TGdesD, TGmatD, TGsrcD, GRID_VOLUME());
        // TensorGrid_CMatrixVector_avx512_expand(TGdesD, TGmatD, TGsrcD, GRID_VOLUME());
#elif defined __AVX__
        TensorGrid_CMatrixVector_avx256(TGdesD, TGmatD, TGsrcD, GRID_VOLUME());
        // TensorGrid_CMatrixVector_avx256_expand(TGdesD, TGmatD, TGsrcD, GRID_VOLUME());
#else
        TensorGrid_CMatrixVector(TGdesD, TGmatD, TGsrcD, GRID_VOLUME());
#endif
        double timeCMVD = watcher.use();

        watcher.reset();
#ifdef __AVX512F__
        TensorGrid_CMatrixVector_avx512(TGdesF, TGmatF, TGsrcF, GRID_VOLUME());
#elif defined __AVX__
        TensorGrid_CMatrixVector_avx256(TGdesF, TGmatF, TGsrcF, GRID_VOLUME());
#else
        TensorGrid_CMatrixVector(TGdesF, TGmatF, TGsrcF, GRID_VOLUME());
#endif
        double timeCMVF = watcher.use();

        auto diffres = diff_vector_norm2(TGdesD, TGdesF, 2 * MAX_ROW * GRID_VOLUME());
        printf("  Gemv(D/F): Acc%6.2lf |time D %8.2e F %8.2e | diff %8.2g | GridSize %ld (L %.0lf)\n",
               timeCMVD / timeCMVF, timeCMVD, timeCMVF, diffres, GRID_VOLUME(), sqrt(sqrt(GRID_VOLUME())));
    }
#endif

    free(TGmatD);
    free(TGsrcD);
    free(TGdesD);

    free(TGmatF);
    free(TGsrcF);
    free(TGdesF);

    return 0;
}