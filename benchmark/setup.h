
#pragma once

#include <cstdlib>
#include <complex>

#ifndef PRECISION
#define PRECISION DOUBLE
#define DataType double
#else
#define PRECISION SINGLE
#define DataType float
#endif

#ifndef MAX_ROW
#define MAX_ROW 3
#endif

#ifndef MAX_COL
#define MAX_COL 3
#endif

typedef DataType *CMatrixPtr[MAX_ROW][MAX_COL][2];
typedef DataType *CVectorPtrCol[MAX_COL][2];
typedef DataType *CVectorPtrRow[MAX_ROW][2];

typedef std::complex<DataType> *ComplexPtr;
typedef std::complex<DataType> Complex;

typedef DataType *RMatrixPtr[MAX_ROW][MAX_COL];
typedef DataType *RVectorPtrCol[MAX_COL];
typedef DataType *RVectorPtrRow[MAX_ROW];

typedef DataType *DataPointer;

#ifndef SIZE
#ifndef LT
#define LT 8
#endif
#ifndef LX
#define LX 8
#endif
#ifndef LY
#define LY 8
#endif
#ifndef LZ
#define LZ 8
#endif
#endif

// #define GRID_VOLUME (LT * LX * LY * LZ)
constexpr size_t GRID_VOLUME()
{
#ifndef SIZE
    return (LT * LX * LY * LZ);
#else
    return SIZE;
#endif
}
