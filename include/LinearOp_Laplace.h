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

#include "TensorGridEnum.h"

void blas_sdgemm_batch();
void blas_ddgemm_batch();
void blas_zdgemm_batch();
void blas_cdgemm_batch();

template <typename FP, Enum_4DGridEvenOdd_t EOnum>
void Laplace(FP* dest, FP* src, int dim, int ThisVolume, int ldx) {
    FP * neighborSrc = src 
}