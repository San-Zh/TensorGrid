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

//首先定义两个辅助宏
#define PRINT_MACRO_HELPER(x) #x
#define PRINT_MACRO(x)        #x "=" PRINT_MACRO_HELPER(x)


#if !defined(SIMD_NO) && !defined(__AVX512F__) && (defined(__AVX__) || defined(__AVX2__))
#define ALIGN_BYTES 32
#define SIMP_OPT    SIMD_AVX256
#include "Simd_avx256.h"
#endif


#if !defined(SIMD_NO) && (defined(__AVX512F__))
#define ALIGN_BYTES 64
#define SIMP_OPT    SIMD_AVX512
#include "Simd_avx512.h"
#endif


/* __ARM_FEATURE_SVE */
#if !defined(SIMD_NO) && (defined(__ARM_FEATURE_SVE))
#define ALIGN_BYTES 64
#define SIMP_OPT    SIMD_ARMSVE
#include "Simd_armsve.h"
#endif


#if defined(SIMD_NO) || !((defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)))
#define ALIGN_BYTES 64
#define SIMP_OPT    SIMD_NO
#include "Simd_common.h"
#endif


#ifdef DEBUG
#pragma message(PRINT_MACRO(SIMD_OPT))
#endif


