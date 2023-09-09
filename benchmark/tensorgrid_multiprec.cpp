
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
#ifdef (HAVE_BLAS&&AOS_LAYOUT)
    std::complex<Tp> alpha(1, 0), beta(0, 0);
    watcher.reset();
    for (size_t v = 0; v < gridSize; v++) {
        cblas_gemv(CblasRowMajor, CblasNoTrans, MAX_COL, MAX_ROW, &alpha, &mat[9 * v], 3, &src[3 * v], 1, &beta,
                    &dest[3 * v], 1);
    }
    double timeCMVD = watcher.use();
    watcher.reset();
    for (size_t v = 0; v < gridSize; v++) {
        cblas_gemv(CblasRowMajor, CblasNoTrans, MAX_COL, MAX_ROW, &alpha, &mat[9 * v], 3, &src[3 * v], 1, &beta,
                    &dest[3 * v], 1);
    }
    double timeCMVF = watcher.use();
#endif

#ifdef AOS_LAYOUT
    watcher.reset();
    ComplexAry_MatrixVector((std::complex<double> *) TGdesD, (std::complex<double> *) TGmatD,
                            (std::complex<double> *) TGsrcD, GRID_VOLUME());
    double timeCMVD = watcher.use();
    //     ComplexAry_MatrixVector_v2(TGdesF, TGmatF, TGsrcF, GRID_VOLUME());
    watcher.reset();
    ComplexAry_MatrixVector((std::complex<float> *) TGdesF, (std::complex<float> *) TGmatF,
                            (std::complex<float> *) TGsrcF, GRID_VOLUME());
    //     ComplexAry_MatrixVector_v2(TGdesD, TGmatD, TGsrcD, GRID_VOLUME());
    double timeCMVF = watcher.use();
#else

#ifdef __AVX512F__
    watcher.reset();
    TensorGrid_CMatrixVector_avx512(TGdesD, TGmatD, TGsrcD, GRID_VOLUME());
    // TensorGrid_CMatrixVector_avx512_expand(TGdesD, TGmatD, TGsrcD, GRID_VOLUME());
    double timeCMVD = watcher.use();
    watcher.reset();
    TensorGrid_CMatrixVector_avx512(TGdesF, TGmatF, TGsrcF, GRID_VOLUME());
    double timeCMVF = watcher.use();
#elif defined __AVX__
    watcher.reset();
    TensorGrid_CMatrixVector_avx256(TGdesD, TGmatD, TGsrcD, GRID_VOLUME());
    double timeCMVD = watcher.use();
    watcher.reset();
    TensorGrid_CMatrixVector_avx256(TGdesF, TGmatF, TGsrcF, GRID_VOLUME());
    double timeCMVF = watcher.use();
#else
    watcher.reset();
    TensorGrid_CMatrixVector(TGdesD, TGmatD, TGsrcD, GRID_VOLUME());
    double timeCMVD = watcher.use();
    watcher.reset();
    TensorGrid_CMatrixVector(TGdesF, TGmatF, TGsrcF, GRID_VOLUME());
    double timeCMVF = watcher.use();
#endif

#endif

    auto diffres = diff_vector_norm2(TGdesD, TGdesF, 2 * MAX_ROW * GRID_VOLUME());
    printf("  Gemv(D/F): Acc%6.2lf |time D %8.2e F %8.2e | diff %8.2g | GridSize %ld (L %.0lf)\n", timeCMVD / timeCMVF,
           timeCMVD, timeCMVF, diffres, GRID_VOLUME(), sqrt(sqrt(GRID_VOLUME())));

    free(TGmatD);
    free(TGsrcD);
    free(TGdesD);

    free(TGmatF);
    free(TGsrcF);
    free(TGdesF);

    return 0;
}