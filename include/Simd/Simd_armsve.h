/**
 * @file Simd_armsve.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-12
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <arm_acle.h>

#if __ARM_FEATURE_SVE_BITS == 512
typedef svint32_t ivec __attribute__((arm_sve_vector_bits(512)));
typedef svbool_t  pred __attribute__((arm_sve_vector_bits(512)));

typedef svint32_t ivec __attribute__((arm_sve_vector_bits(512)));
typedef svbool_t  pred __attribute__((arm_sve_vector_bits(512)));

svint32_t g1; // Invalid, svint32_t is sizeless
ivec      g2; // OK
svbool_t  g3; // Invalid, svbool_t is sizeless
pred      g4; // OK


// clang-format off
struct wrap1 { svint32_t x; }; // Invalid, svint32_t is sizeless
struct wrap2 { ivec      x; }; // OK
struct wrap3 { svbool_t  x; }; // Invalid, svbool_t is sizeless
struct wrap4 { ipred     x; }; // OK
// clang-format on

size_t size1 = sizeof(svint32_t); // Invalid, svint32_t is sizeless
size_t size2 = sizeof(ivec);      // OK, equals 64
size_t size3 = sizeof(svbool_t);  // Invalid, svbool_t is sizeless
size_t size4 = sizeof(ipred);     // OK, equals 8

#endif