/**
 * @file transfer.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-14
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include <iostream>

template <typename Tp>
void tranfer2TG(Tp *TGdest, Tp *src, size_t sizeTensor, size_t sizeGrid)
{
    for (size_t v = 0; v < sizeGrid; v++) {
        for (size_t its = 0; its < sizeTensor; its++) {
            *(TGdest + its * sizeGrid + v) = *(src + v * sizeTensor + its);
        }
    }
    // Tp *tgptr[sizeTensor];
    // for (size_t it = 0; it < sizeTensor; it++) {
    //     tgptr[it] = TGdest + it * sizeGrid;
    // }
    // for (size_t v = 0; v < sizeGrid; v++) {
    //     for (size_t its = 0; its < sizeTensor; its++) {
    //         *(tgptr[its] + v) = *(src + v * sizeTensor + its);
    //         // tgptr[its][v] = src[v * sizeTensor + its];
    //     }
    // }
}


/**
 * @brief \todo Wrong |src - TGsrc| !=0
 * 
 * @tparam Tp 
 * @param dest 
 * @param TGsrc 
 * @param sizeTensor 
 * @param sizeGrid 
 */
template <typename Tp>
void tranfer2general(Tp *dest, Tp *TGsrc, size_t sizeTensor, size_t sizeGrid)
{
    for (size_t v = 0; v < sizeGrid; v++) {
        for (size_t its = 0; its < sizeTensor; its++) {
            *(dest + v * sizeTensor + its) = *(TGsrc + its * sizeGrid + v);
        }
    }
}

#if 0
template <typename Tp>
void tranfer2TG(Tp *TGdest, Tp *src, size_t sizeTensor, size_t sizeGrid)
{
    Tp *srcptr;
    Tp *tgptr[sizeTensor];
    for (size_t it = 0; it < sizeTensor; it++) {
        tgptr[it] = &((TGdest + it * sizeGrid)[0]);
        // tgptr[it] = TGdest + it * sizeGrid;
    }

    for (srcptr = &src[0]; srcptr < src + sizeTensor * sizeGrid; srcptr += sizeTensor) {
        for (size_t it = 0; it < sizeTensor; it++) {
            *(tgptr[it]) = *(srcptr + it);
        }
        for (size_t it = 0; it < sizeTensor; it++) {
            tgptr[it]++;
        }
    }
}

/**
 * @brief 
 * 
 * @tparam Tp 
 * @param dest 
 * @param TGsrc 
 * @param sizeTensor 
 * @param sizeGrid 
 */
template <typename Tp>
void tranfer2general(Tp *dest, Tp *TGsrc, size_t sizeTensor, size_t sizeGrid)
{
    Tp *desptr;
    Tp *tgptr[sizeTensor];
    for (size_t it = 0; it < sizeTensor; it++) {
        tgptr[it] = &((TGsrc + it * sizeGrid)[0]);
    }

    for (desptr = &dest[0]; desptr < dest + sizeTensor * sizeGrid; desptr += sizeTensor) {
        for (size_t it = 0; it < sizeTensor; it++) {
            *(desptr + it) = *(tgptr[it]);
        }
        for (size_t it = 0; it < sizeTensor; it++) {
            tgptr[it]++;
        }
    }
}

#endif

template <typename Tp>
void random(Tp *src, size_t size)
{
    Tp RdmInv = (Tp) (1) / static_cast<Tp>(RAND_MAX);
    for (size_t i = 0; i < size; i++) {
        src[i] = static_cast<Tp>(random()) * RdmInv;
        // src[i] = static_cast<DataType>(i);
    }
}

template <typename Tp>
Tp diff_Ary_TGAry(Tp *A, Tp *TGA, size_t sizeTensor, size_t sizeGrid)
{
    Tp *pb[sizeTensor];
    for (size_t its = 0; its < sizeTensor; its++) { pb[its] = TGA + its * sizeGrid; }

    Tp diff = 0.0;
    Tp sum  = 0.0;
    // for (size_t i = 0; i < 4; i++){
    for (size_t i = 0; i < sizeGrid; i++) {
        for (size_t its = 0; its < sizeTensor; its++) {
            diff = A[i * sizeTensor + its] - pb[its][i];
            sum += diff;
#ifdef DEBUG_PRINT
            printf("%14.4e%14.4e%14.4e\n", A[i * sizeTensor + its], pb[its][i], diff);
#endif
        }
    }
    return sum;
}

template <typename Tp1, typename Tp2>
Tp1 diff_vector_norm2(Tp1 *A, Tp2 *B, size_t Size)
{
    Tp1 *ptrA;
    Tp2 *ptrB;
    Tp1  diff = 0.0;
    Tp2  res  = 0.0;
    // for (size_t i = 0; i < 4; i++){
    for (ptrA = A, ptrB = B; ptrA != A + Size; ptrA++, ptrB++) {
        diff = *ptrA - *ptrB;
        res += diff * diff;
#ifdef DEBUG_PRINT
        printf("%14.4e%14.4e%14.4e\n", *ptrA, *ptrB, res);
#endif
    }
    return res;
}

template <typename Tp1, typename Tp2>
void xeqy(Tp1 *A, Tp2 *B, size_t Size)
{
    Tp1 *ptrA;
    Tp2 *ptrB;
    for (ptrA = A, ptrB = B; ptrA != A + Size; ptrA++, ptrB++) { *ptrA = *ptrB; }
}