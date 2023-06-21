
#pragma once

#include <cstdlib>
#include <complex>
#include "setup.h"
#include "transfer.h"

template <typename Tp>
void AryIO(Tp *dest, Tp *src, size_t size)
{
    for (size_t v = 0; v < size; v++) {
        dest[v] = src[v];
    }
}

template <typename Tp>
void AryRead(Tp *src, size_t size)
{
    Tp tmp;
    for (size_t v = 0; v < size; v++) {
        tmp = src[v];
    }
}

template <typename Tp>
void AryWrite(Tp *dest, size_t size)
{
    for (size_t v = 0; v < size; v++) {
        dest[v] = 1.0;
    }
}

void ComplexAry_CXYpY(ComplexPtr dest, ComplexPtr X, ComplexPtr Y, size_t size)
{
    for (size_t v = 0; v < size; v++) {
        dest[v] = X[v] * Y[v] + Y[v];
    }
}

void TensorGrid_CXYpY(DataType *dest, DataType *X, DataType *Y, size_t tensorSize, size_t gridSize)
{
    DataType re, im;
    for (size_t its = 0; its < tensorSize; its++) {
        DataType *Xre = X + its * 2 * gridSize;
        DataType *Xim = X + (its * 2 + 1) * gridSize;
        DataType *Yre = Y + its * 2 * gridSize;
        DataType *Yim = Y + (its * 2 + 1) * gridSize;
        DataType *destre = dest + its * 2 * gridSize;
        DataType *destim = dest + (its * 2 + 1) * gridSize;
        for (size_t v = 0; v < gridSize; v++) {
            re = Xre[v] * Yre[v] - Xim[v] * Yim[v] + Yre[v];
            im = Xre[v] * Yim[v] + Xim[v] * Yre[v] + Yim[v];
            destre[v] = re;
            destim[v] = im;
        }
    }
}

void ComplexAry_CXTY(ComplexPtr dest, ComplexPtr X, ComplexPtr Y, size_t tensorSize, size_t gridSize)
{
    for (size_t v = 0; v < gridSize; v++) {
        for (size_t its = 0; its < tensorSize; its++) {
            dest[v] += X[its + v * tensorSize] * Y[its + v * tensorSize];
        }
    }
}

void TensorGrid_CXTY(DataType *dest, DataType *X, DataType *Y, size_t tensorSize, size_t gridSize)
{
    DataType re, im;
    for (size_t its = 0; its < tensorSize; its++) {
        DataType *Xre = X + (its * 2) * gridSize;
        DataType *Xim = X + (its * 2 + 1) * gridSize;
        DataType *Yre = Y + (its * 2) * gridSize;
        DataType *Yim = Y + (its * 2 + 1) * gridSize;
        DataType *destre = dest;
        DataType *destim = dest + gridSize;
        for (size_t v = 0; v < gridSize; v++) {
            re = Xre[v] * Yre[v] - Xim[v] * Yim[v];
            im = Xre[v] * Yim[v] + Xim[v] * Yre[v];
            destre[v] += re;
            destim[v] += im;
        }
    }

    // DataPointer Xre[tensorSize], Xim[tensorSize];
    // DataPointer Yre[tensorSize], Yim[tensorSize];
    // DataPointer destre = dest;
    // DataPointer destim = dest + gridSize;
    // for (size_t its = 0; its < tensorSize; its++) {
    //     Xre[its] = X + (its * 2) * gridSize;
    //     Xim[its] = X + (its * 2 + 1) * gridSize;
    //     Yre[its] = Y + (its * 2) * gridSize;
    //     Yim[its] = Y + (its * 2 + 1) * gridSize;
    //     // }
    //     // for (size_t its = 0; its < tensorSize; its++) {
    //     for (size_t v = 0; v < gridSize; v++) {
    //         re = Xre[its][v] * Yre[its][v] - Xim[its][v] * Yim[its][v];
    //         im = Xre[its][v] * Yim[its][v] + Xim[its][v] * Yre[its][v];
    //         destre[v] += re;
    //         destim[v] += im;
    //     }
    // }
}

// void TensorGrid_CXYpY(DataType *dest, DataType *X, DataType *Y, size_t tensorSize, size_t gridSize)
// {
//     DataPointer Xre[tensorSize];
//     DataPointer Xim[tensorSize];
//     DataPointer Yre[tensorSize];
//     DataPointer Yim[tensorSize];
//     DataPointer Desre[tensorSize];
//     DataPointer Desim[tensorSize];
//     for (size_t it = 0; it < tensorSize; it++) {
//         Xre[it] = X + it * 2 * gridSize;
//         Xim[it] = X + (it * 2 + 1) * gridSize;

//         Yre[it] = Y + it * 2 * gridSize;
//         Yim[it] = Y + (it * 2 + 1) * gridSize;

//         Desre[it] = dest + it * 2 * gridSize;
//         Desim[it] = dest + (it * 2 + 1) * gridSize;
//     }

//     for (size_t it = 0; it < tensorSize; it++) {
//         for (size_t v = 0; v < gridSize; v++) {
//             DataType re = Xre[it][v] * Yre[it][v] - Xim[it][v] * Yim[it][v] + Yre[it][v];
//             DataType im = Xre[it][v] * Yim[it][v] + Xim[it][v] * Yre[it][v] + Yim[it][v];
//             Desre[it][v] = re;
//             Desim[it][v] = im;
//         }
//     }
// }

/// @brief when gridSize > 1024*8 double, this method accelerate rate mothan 4.0, compare to the method as followed;
///         here test pc L1(32K+32K) L2(1024K) L3(17M)
/// @param dest
/// @param mat
/// @param src
/// @param gridSize
template <typename Tp>
void ComplexAry_MatrixVector(std::complex<Tp> *dest, std::complex<Tp> *mat, std::complex<Tp> *src, size_t gridSize)
{
    for (size_t v = 0; v < gridSize; v++) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            for (size_t col = 0; col < MAX_COL; col++) {
                dest[v * MAX_ROW + row] += mat[v * MAX_ROW * MAX_COL + row * MAX_COL + col] * src[v * MAX_COL + col];
            }
        }
    }
}

void ComplexAry_MatrixVector(double *dest, double *mat, double *src, size_t gridSize)
{
    for (size_t v = 0; v < gridSize; v++) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            double res_re = 0.0;
            double res_im = 0.0;
            double vsrc[MAX_COL][2];
            for (size_t col = 0; col < MAX_COL; col++) {
                vsrc[col][0] = src[v * 2 * MAX_COL + 2 * col + 0];
                vsrc[col][1] = src[v * 2 * MAX_COL + 2 * col + 1];
            }
            for (size_t col = 0; col < MAX_COL; col++) {
                double mat_re = mat[v * 2 * MAX_ROW * MAX_COL + row * 2 * MAX_COL + 2 * col + 0];
                double mat_im = mat[v * 2 * MAX_ROW * MAX_COL + row * 2 * MAX_COL + 2 * col + 1];
                res_re += mat_re * vsrc[col][0] - mat_im * vsrc[col][1];
                res_im += mat_re * vsrc[col][1] + mat_im * vsrc[col][0];
            }
            dest[v * 2 * MAX_ROW + 2 * row + 0] = res_re;
            dest[v * 2 * MAX_ROW + 2 * row + 1] = res_im;
        }
    }
}

void ComplexAry_MatrixVector(float *dest, float *mat, float *src, size_t gridSize)
{
    for (size_t v = 0; v < gridSize; v++) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            float res_re = 0.0;
            float res_im = 0.0;
            float vsrc[MAX_COL][2];
            for (size_t col = 0; col < MAX_COL; col++) {
                vsrc[col][0] = src[v * 2 * MAX_COL + 2 * col + 0];
                vsrc[col][1] = src[v * 2 * MAX_COL + 2 * col + 1];
            }
            for (size_t col = 0; col < MAX_COL; col++) {
                float mat_re = mat[v * 2 * MAX_ROW * MAX_COL + row * 2 * MAX_COL + 2 * col + 0];
                float mat_im = mat[v * 2 * MAX_ROW * MAX_COL + row * 2 * MAX_COL + 2 * col + 1];
                res_re += mat_re * vsrc[col][0] - mat_im * vsrc[col][1];
                res_im += mat_re * vsrc[col][1] + mat_im * vsrc[col][0];
            }
            dest[v * 2 * MAX_ROW + 2 * row + 0] = res_re;
            dest[v * 2 * MAX_ROW + 2 * row + 1] = res_im;
        }
    }
}

void ComplexAry_MatrixVector02(ComplexPtr dest, ComplexPtr mat, ComplexPtr src, size_t gridSize)
{
    /// the lager grid volume is, the lower performance than the base ComplexAry_MatrixVector()
    // for (size_t row = 0; row < MAX_ROW; row++) {
    //     for (size_t col = 0; col < MAX_COL; col++) {
    //         for (size_t v = 0; v < gridSize; v++) {
    //             dest[v * MAX_ROW + row] += mat[v * MAX_ROW * MAX_COL + row * MAX_COL + col] * src[v * MAX_COL + col];
    //         }
    //     }
    // }
    for (size_t v = 0; v < gridSize; v++) {
        Complex res[MAX_ROW] = {0.0};
        for (size_t row = 0; row < MAX_ROW; row++) {
            for (size_t col = 0; col < MAX_COL; col++) {
                res[row] += mat[v * MAX_ROW * MAX_COL + row * MAX_COL + col] * src[v * MAX_COL + col];
            }
            dest[v * MAX_ROW + row] = res[row];
        }
    }
}

template <typename Tp>
void TensorGrid_CMatrixVector(Tp *dest, Tp *mat, Tp *src, size_t gridSize)
{
    Tp *pd[MAX_ROW][2];
    Tp *ps[MAX_COL][2];
    Tp *pm[MAX_ROW][MAX_COL][2];
    // CVectorPtrRow pd;
    // CVectorPtrRow ps;
    // CMatrixPtr pm;

    for (size_t col = 0; col < MAX_COL; col++) {
        ps[col][0] = src + col * 2 * gridSize;
        ps[col][1] = src + (col * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < MAX_ROW; row++) {
        pd[row][0] = dest + row * 2 * gridSize;
        pd[row][1] = dest + (row * 2 + 1) * gridSize;
    }
    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            pm[row][col][0] = mat + (row * 2 * MAX_COL + 2 * col) * gridSize;
            pm[row][col][1] = mat + (row * 2 * MAX_COL + 2 * col + 1) * gridSize;
        }
    }

    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            for (size_t v = 0; v < gridSize; v++) {
                Tp re = pm[row][col][0][v] * ps[col][0][v] - pm[row][col][1][v] * ps[col][1][v];
                Tp im = pm[row][col][0][v] * ps[col][1][v] + pm[row][col][1][v] * ps[col][0][v];
                pd[row][0][v] += re;
                pd[row][1][v] += im;
                // *(pd[row][0] + v) += *(pm[row][col][0] + v) * *(ps[col][0] + v) - *(pm[row][col][1] + v) * *(ps[col][1] + v);
                // *(pd[row][1] + v) += *(pm[row][col][0] + v) * *(ps[col][1] + v) + *(pm[row][col][1] + v) * *(ps[col][0] + v);
            }
        }
    }
}

#if (MAX_ROW == 3) && (MAX_COL == 3)
void TensorGrid_CMatrixVector02(DataType *TGdes, DataType *TGmat, DataType *TGsrc, size_t gridSize)
{
    DataPointer m00re = TGmat + (0 * 2 * MAX_COL + 0) * gridSize;
    DataPointer m00im = TGmat + (0 * 2 * MAX_COL + 1) * gridSize;
    DataPointer m01re = TGmat + (0 * 2 * MAX_COL + 2) * gridSize;
    DataPointer m01im = TGmat + (0 * 2 * MAX_COL + 3) * gridSize;
    DataPointer m02re = TGmat + (0 * 2 * MAX_COL + 4) * gridSize;
    DataPointer m02im = TGmat + (0 * 2 * MAX_COL + 5) * gridSize;
    DataPointer m10re = TGmat + (1 * 2 * MAX_COL + 0) * gridSize;
    DataPointer m10im = TGmat + (1 * 2 * MAX_COL + 1) * gridSize;
    DataPointer m11re = TGmat + (1 * 2 * MAX_COL + 2) * gridSize;
    DataPointer m11im = TGmat + (1 * 2 * MAX_COL + 3) * gridSize;
    DataPointer m12re = TGmat + (1 * 2 * MAX_COL + 4) * gridSize;
    DataPointer m12im = TGmat + (1 * 2 * MAX_COL + 5) * gridSize;
    DataPointer m20re = TGmat + (2 * 2 * MAX_COL + 0) * gridSize;
    DataPointer m20im = TGmat + (2 * 2 * MAX_COL + 1) * gridSize;
    DataPointer m21re = TGmat + (2 * 2 * MAX_COL + 2) * gridSize;
    DataPointer m21im = TGmat + (2 * 2 * MAX_COL + 3) * gridSize;
    DataPointer m22re = TGmat + (2 * 2 * MAX_COL + 4) * gridSize;
    DataPointer m22im = TGmat + (2 * 2 * MAX_COL + 5) * gridSize;

    DataPointer vs0re = TGsrc + 0 * gridSize;
    DataPointer vs0im = TGsrc + 1 * gridSize;
    DataPointer vs1re = TGsrc + 2 * gridSize;
    DataPointer vs1im = TGsrc + 3 * gridSize;
    DataPointer vs2re = TGsrc + 4 * gridSize;
    DataPointer vs2im = TGsrc + 5 * gridSize;

    DataPointer vd0re = TGdes + 0 * gridSize;
    DataPointer vd0im = TGdes + 1 * gridSize;
    DataPointer vd1re = TGdes + 2 * gridSize;
    DataPointer vd1im = TGdes + 3 * gridSize;
    DataPointer vd2re = TGdes + 4 * gridSize;
    DataPointer vd2im = TGdes + 5 * gridSize;

    for (size_t i = 0; i < gridSize; i++) {
        vd0re[i] = m00re[i] * vs0re[i] - m00im[i] * vs0im[i] + m01re[i] * vs1re[i] - m01im[i] * vs1im[i] +
                   m02re[i] * vs2re[i] - m02im[i] * vs2im[i];
        vd0im[i] = m00re[i] * vs0im[i] + m00im[i] * vs0re[i] + m01re[i] * vs1im[i] + m01im[i] * vs1re[i] +
                   m02re[i] * vs2im[i] + m02im[i] * vs2re[i];
        vd1re[i] = m10re[i] * vs0re[i] - m10im[i] * vs0im[i] + m11re[i] * vs1re[i] - m11im[i] * vs1im[i] +
                   m12re[i] * vs2re[i] - m12im[i] * vs2im[i];
        vd1im[i] = m10re[i] * vs0im[i] + m10im[i] * vs0re[i] + m11re[i] * vs1im[i] + m11im[i] * vs1re[i] +
                   m12re[i] * vs2im[i] + m12im[i] * vs2re[i];
        vd2re[i] = m20re[i] * vs0re[i] - m20im[i] * vs0im[i] + m21re[i] * vs1re[i] - m21im[i] * vs1im[i] +
                   m22re[i] * vs2re[i] - m22im[i] * vs2im[i];
        vd2im[i] = m20re[i] * vs0im[i] + m20im[i] * vs0re[i] + m21re[i] * vs1im[i] + m21im[i] * vs1re[i] +
                   m22re[i] * vs2im[i] + m22im[i] * vs2re[i];
    }
}
#endif

/**
 * @brief this metod has lower performancee than above
 * 
 * @param dest 
 * @param mat 
 * @param src 
 * @param gridSize 
 */
void TensorGrid_CMatrixVector(CVectorPtrRow dest, CMatrixPtr mat, CVectorPtrCol src, size_t gridSize)
{
    DataType re;
    DataType im;
    for (size_t row = 0; row < MAX_ROW; row++) {
        for (size_t col = 0; col < MAX_COL; col++) {
            for (size_t v = 0; v < gridSize; v++) {
                re = mat[row][col][0][v] * src[col][0][v] - mat[row][col][1][v] * src[col][1][v];
                im = mat[row][col][0][v] * src[col][1][v] + mat[row][col][1][v] * src[col][0][v];
                dest[row][0][v] += re;
                dest[row][1][v] += im;
            }
        }
    }
}

#if 0

/**
 * @brief this metod has lower performancee than above
 * 
 * @param dest 
 * @param mat 
 * @param src 
 * @param gridSize 
 */
void TensorGrid_CMatrixVector03(CVectorPtrRow dest, CMatrixPtr mat, CVectorPtrCol src, size_t gridSize)
{
    DataType re;
    DataType im;
    size_t Wsmd = (64 / sizeof(DataType));
    size_t Nsmd = gridSize / Wsmd;
    // printf("Grid size: %ld   Wsmd * Nsmd = %ld * %ld = %ld\n", gridSize, Wsmd, Nsmd, Wsmd * Nsmd);
    for (size_t i = 0; i < Nsmd; i++) {
        for (size_t row = 0; row < MAX_ROW; row++) {
            for (size_t col = 0; col < MAX_COL; col++) {
                // for (size_t v = 0; v < gridSize; v++) {
                for (size_t vi = 0; vi < Wsmd; vi++) {
                    size_t v = i * Wsmd + vi;
                    re = mat[row][col][0][v] * src[col][0][v] - mat[row][col][1][v] * src[col][1][v];
                    im = mat[row][col][0][v] * src[col][1][v] + mat[row][col][1][v] * src[col][0][v];
                    dest[row][0][v] += re;
                    dest[row][1][v] += im;
                }
            }
        }
    }
    // for (size_t row = 0; row < MAX_ROW; row++) {
    //     for (size_t v = 0; v < gridSize; v++) {
    //         dest[row][0][v] += mat[row][0][0][v] * src[0][0][v] - mat[row][0][1][v] * src[0][1][v];
    //         dest[row][1][v] += mat[row][0][0][v] * src[0][1][v] + mat[row][0][1][v] * src[0][0][v];
    //         dest[row][0][v] += mat[row][1][0][v] * src[1][0][v] - mat[row][1][1][v] * src[1][1][v];
    //         dest[row][1][v] += mat[row][1][0][v] * src[1][1][v] + mat[row][1][1][v] * src[1][0][v];
    //         dest[row][0][v] += mat[row][2][0][v] * src[2][0][v] - mat[row][2][1][v] * src[2][1][v];
    //         dest[row][1][v] += mat[row][2][0][v] * src[2][1][v] + mat[row][2][1][v] * src[2][0][v];
    //     }
    // }
#endif