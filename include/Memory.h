/**
 * @file Memory.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-02-27
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <x86intrin.h>
#include <stdlib.h>
#include "MemoryEnum.h"

#define ALIGN_BYTES 32

/**
 * @brief 
 * 
 * @param addr 
 * @param bytes 
 * @param memLevel 
 */
void MemoryAlloc(void *addr, size_t bytes, MemLevel_t memLevel = MemLevel0)
{
    if (memLevel == MemLevel0) {

#if defined _SIMD_
        addr = _mm_malloc(bytes, ALIGN_BYTES);
#else
        addr = malloc(bytes);
#endif

    } else if (memLevel = MemLevel1) {

#ifdef _HIP_
        dcuErrchk(hipHostMalloc(&addr, bytes));
#endif
    }
}

/**
 * @brief 
 * 
 * @param addr 
 */
void MemoryFree(void *addr, MemLevel_t memLevel = MemLevel0)
{
    if (memLevel == MemLevel0) {
#ifdef _SIMD_
        _mm_free(addr);
#else
        free(addr);
#endif
    } else if (memLevel = MemLevel1) {
#ifdef _HIP_
        dcuErrchk(hipHostMalloc(&addr, bytes));
#endif
    }
}