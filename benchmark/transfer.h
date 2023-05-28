
#pragma once

#include <cstdlib>
#include <complex>

template <typename Tp>
void tranfer2TG(Tp *dest, Tp *src, size_t sizeTensor, size_t sizeGrid)
{
    for (size_t v = 0; v < sizeGrid; v++) {
        for (size_t its = 0; its < sizeTensor; its++) {
            *(dest + its * sizeGrid + v) = *(src + v * sizeTensor + its);
        }
    }
}

template <typename Tp>
Tp check_diff(Tp *A, Tp *B, size_t sizeTensor, size_t sizeGrid)
{
    Tp *pb[sizeTensor];
    for (size_t its = 0; its < sizeTensor; its++) {
        pb[its] = B + its * sizeGrid;
    }

    Tp diff = 0.0;
    Tp sum = 0.0;
    // for (size_t i = 0; i < 4; i++){
    for (size_t i = 0; i < sizeGrid; i++) {
        for (size_t its = 0; its < sizeTensor; its++) {
            diff = A[i * sizeTensor + its] - pb[its][i];
            sum += diff;
#ifdef debug
            // printf("%14.4e%14.4e%14.4e\n", A[i * sizeTensor + its], pb[its][i], diff);
#endif
        }
    }
#ifdef debug
    // printf("%42s\n", "--------------------------");
    printf("%28s%14.4e\n\n", "sum of diff", sum);
#endif
    return sum;
}
