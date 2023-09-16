/**
 * @file SimdTupleTypes.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-13
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "SimdTraits.h"

template <class vtype, unsigned _N>
class iVector {
  public:
    vtype       &operator[](unsigned _n) { return vec[_n]; }
    const vtype &operator[](unsigned _n) const { return vec[_n]; }
    enum { Size = _N };

  private:
    vtype vec[_N];
};

// template <class vtype, unsigned _M, unsigned _N = _M>
// class iMatrix {
//   public:
//     typedef typename SimdTraits<vtype>::vtype vtype;
//     typedef typename SimdTraits<vtype>::dtype dtype;

//   public:
//     // vtype       &operator[][](unsigned _m, unsigned _n) { return mat[_m][_n]; }
//     // const vtype &operator[][](unsigned _m, unsigned _n) const { return mat[_m][_n]; }
//     enum {};

//   private:
//     vtype _mat[_M][_N];
// };

// template <class vtype, int M, int N = M>
// class iMatrix {
//   public:
//     typename SimdTraits<vtype>::value_type mat[M][N];
// };


template <class vtype, int N>
using kerVector = typename SimdTraits<vtype>::vtype[N];


template <class vtype, unsigned M, unsigned N>
using kerMatrix = typename SimdTraits<vtype>::vtype[M][N];


// template <class iVect, unsigned _N>
// void iVectorLoad(iVect &A, const typename iVect::ptr_type _p)
// {
//     for (unsigned i = 0; i < _N; i++) { SimdLoad(A[i], _p[i]); }
// }


template <class vtype, unsigned _N>
static inline void iVectorLoad(iVector<vtype, _N> &A, typename SimdTraits<vtype>::ptr_type *_p)
{
    for (unsigned i = 0; i < _N; i++) { SimdLoad(A[i], _p[i]); }
}
