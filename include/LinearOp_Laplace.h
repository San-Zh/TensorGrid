/**
 * @file LinearOp_Laplace.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-08-31
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "TensorGridClass.h"
#include "TensorGridEnum.h"
#include "cstdlib"

template <typename TF, int GridDir>
TF *shift_offset(const TF *in, int offset)
{
    return &in[offset];
}

// void blas_sdgemm_batch(void * dest, void *src, bool dag);
// void blas_ddgemm_batch(void * dest, void *src, bool dag);
// void blas_zdgemm_batch(void * dest, void *src, bool dag);
// void blas_cdgemm_batch(void * dest, void *src, bool dag);

template <typename FP, EnumBit_4DGridEvenOdd_t EOGridEnum, EnumBit_4DGridDir_t DirEnum>
void Laplace(FP *dest, FP *src, int Volume1Blk, int LenDir);

template <typename FP, EnumBit_4DGridEvenOdd_t EOGridEnum>
void C_Laplace_x(FP *dest, const FP *Ub, const FP *Uf, FP *src, unsigned int M, unsigned int N, unsigned int K,
                 const size_t GridVolume, int LenX)
{
    size_t stride = GridVolume * M * K * 2;
    FP *pUb[M][K][2];
    for (size_t i = 0; i < count; i++) {
      
    }

    FP *pUf[M][K][2];
    FP *pFf[K][N][2];
    FP *pFb[K][N][2];

    for (size_t i = 0; i < V; i++) {
    }
}