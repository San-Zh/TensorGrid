/**
 * @file Simd_opt.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-12
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

// #if (not defined(SIMD_OPT)) || (SIMD_OPT == SIMD_OPT) || (SIMD_OPT == 1)
// // #endif
// #if defined(SIMD_NO) || (SIMD_OPT == SIMD_NO)
// #include "Simd_common.h"
// #endif

#if defined(__AVX512F__) || (SIMD_OPT == AVX512)
#define SIMD_YES
#include "Simd_avx512.h"
#endif


#if defined(__AVX__) || (SIMD_OPT == AVX256)
#define SIMD_YES
#include "Simd_avx256.h"
#endif


/* __ARM_FEATURE_SVE */
#ifdef __ARM_FEATURE_SVE
#include "Simd_armsve.h"
#endif




//首先定义两个辅助宏
#define PRINT_MACRO_HELPER(x) #x
#define PRINT_MACRO(x)        #x "=" PRINT_MACRO_HELPER(x)

// #ifdef DEBUG
// #pragma message(PRINT_MACRO(SIMD_OPT))
// #endif