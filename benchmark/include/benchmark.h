/**
 * @file benchmark.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-14
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include "benchmark_base_generic.h"

#include "benchmark_TensorGridBlas_autosimd.h"

#if defined(__AVX__) || defined(__AVX2__)
#include "benchmark_TensorGridBlas_avx256.h"
#endif

#if defined(__AVX512F__)
#include "benchmark_TensorGridBlas_avx512.h"
#endif
