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

#pragma once

#include "SimdTraits.h"

/**
 * @brief iVerctor<vtype, N>, vector of SimdTupleTypes
 * 
 * @tparam vtype 
 * @tparam _N 
 */
template <class vtype, unsigned _N>
class iVector {
  public:
    typedef typename SimdTraits<vtype>::ptr_type ptype[_N];

    inline vtype       &operator()(unsigned _n) { return _vec[_n]; }
    inline const vtype &operator()(unsigned _n) const { return _vec[_n]; }

    inline void load(ptype _p)
    {
        for (unsigned _n = 0; _n < _N; _n++) { _vec[_n].load(_p[_n]); }
    };

    inline void load(ptype _p, const size_t _ofs)
    {
        for (unsigned _n = 0; _n < _N; _n++) { _vec[_n].load(_p[_n], _ofs); }
    };

    inline void store(ptype _p)
    {
        for (unsigned _n = 0; _n < _N; _n++) { _vec[_n].load(_p[_n]); }
    };

    inline void store(ptype _p, const size_t _ofs)
    {
        for (unsigned _n = 0; _n < _N; _n++) { _vec[_n].store(_p[_n], _ofs); }
    };

    inline void setzero()
    {
        for (unsigned _n = 0; _n < _N; _n++) { _vec[_n].setzero(); }
    }

  private:
    vtype _vec[_N];
};


/**
 * @brief iMatrix<vtype, N>, Matrix of SimdTupleTypes
 * 
 * @tparam vtype 
 * @tparam _N 
 * @param A 
 * @param _p 
 */
template <class vtype, unsigned _N>
static inline void iVectorLoad(iVector<vtype, _N> &A, typename SimdTraits<vtype>::ptr_type *_p)
{
    for (unsigned i = 0; i < _N; i++) { SimdLoad(A[i], _p[i]); }
}



template <class vtype, unsigned _M, unsigned _N>
class iMatrix {
  public:
    typedef typename SimdTraits<vtype>::ptr_type ptype[_M][_N];

    vtype       &operator()(unsigned _m, unsigned _n) { return _mat[_m][_n]; }
    const vtype &operator()(unsigned _m, unsigned _n) const { return _mat[_m][_n]; }

    vtype       &vec(unsigned _m, unsigned _n) { return _mat[_m][_n]; }
    const vtype &vec(unsigned _m, unsigned _n) const { return _mat[_m][_n]; }

    inline void setzero()
    {
        for (unsigned _m = 0; _m < _M; _m++) {
            for (unsigned _n = 0; _n < _N; _n++) { _mat[_m][_n].setzero(); }
        }
    }


    inline void load(ptype _p)
    {
        for (unsigned _m = 0; _m < _M; _m++) {
            for (unsigned _n = 0; _n < _N; _n++) { _mat[_m][_n].load(_p[_m][_n]); }
        }
    };

    inline void load(ptype _p, const size_t _ofs)
    {
        for (unsigned _m = 0; _m < _M; _m++) {
            for (unsigned _n = 0; _n < _N; _n++) { _mat[_m][_n].laod(_p[_m][_n], _ofs); }
        }
    };

    inline void store(ptype _p)
    {
        for (unsigned _m = 0; _m < _M; _m++) {
            for (unsigned _n = 0; _n < _N; _n++) { _mat[_m][_n].store(_p[_m][_n]); }
        }
    };

    inline void store(ptype _p, const size_t _ofs)
    {
        for (unsigned _m = 0; _m < _M; _m++) {
            for (unsigned _n = 0; _n < _N; _n++) { _mat[_m][_n].store(_p[_m][_n], _ofs); }
        }
    };


  private:
    typename SimdTraits<vtype>::vtype _mat[_M][_N];
};
