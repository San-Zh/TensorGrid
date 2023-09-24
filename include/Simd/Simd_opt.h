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


// clang-format off

#if defined(SIMD_NO)
    #define ALIGN_BYTES 64
    #define SIMP_OPT    SIMD_NO
    #include "Simd_common.h"

#else
    #if (defined(__ARM_FEATURE_SVE))
        #define ALIGN_BYTES 64
        #define SIMP_OPT    SIMD_ARMSVE
        #include "Simd_armsve.h"
    
    #else

        #if !defined(SIMD_NO) && (defined(__AVX512F__))
            #define ALIGN_BYTES 64
            #define SIMP_OPT    SIMD_AVX512
            #include "Simd_avx512.h"

        #else
            #if (defined(__AVX__) || defined(__AVX2__))
                #define ALIGN_BYTES 32
                #define SIMP_OPT    SIMD_AVX256
                #include "Simd_avx256.h"

            #else
                #if (defined(__SSE__))
                    #define ALIGN_BYTES 32
                    #define SIMP_OPT    SIMD_AVX256
                    #include "Simd_sse.h"

                #else
                    #define ALIGN_BYTES 64
                    #define SIMP_OPT    SIMD_NO
                    #include "Simd_common.h"

                #endif

            #endif

        #endif

    #endif

#endif


// clang-format on


/* __ARM_FEATURE_SVE */



// #ifdef DEBUG
// #pragma message(PRINT_MACRO(SIMD_OPT))
// #endif
