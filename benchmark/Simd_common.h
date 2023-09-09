/**
 * @file Simd_common.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-09
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#if not defined(ENABLE_SIMD)
struct vRealF {
    float v[16];
};

struct vRealD {
    double v[8];
};

#define _for(i, _N, e) \
    for (int i = 0; i < _N; i++) { e; }

// load
static inline vRealF SimdLoad(const float *_p)
{
    vRealF ret;
    _for(i, 16, ret.v[i] = _p[i]);
    return ret;
}

static inline vRealD SimdLoad(const double *_p)
{
    vRealD ret;
    _for(i, 8, ret.v[i] = _p[i]);
    return ret;
}

// store
static inline void SimdStore(float *_p, vRealF a)
{
    vRealF ret;
    _for(i, 16, _p[i] = a.v[i]);
}
static inline void SimdStore(double *_p, vRealD a)
{
    vRealD ret;
    _for(i, 8, _p[i] = a.v[i]);
}

// add
static inline vRealF SimdAdd(vRealF a, vRealF b)
{
    vRealF ret;
    _for(i, 16, ret.v[i] = a.v[i] + b.v[i]);
    return ret;
}
static inline vRealD SimdAdd(vRealD a, vRealD b)
{
    vRealD ret;
    _for(i, 8, ret.v[i] = a.v[i] + b.v[i]);
}

// sub
static inline vRealF SimdSub(vRealF a, vRealF b)
{
    vRealF ret;
    _for(i, 16, ret.v[i] = a.v[i] - b.v[i]);
    return ret;
}
static inline vRealD SimdAdd(vRealD a, vRealD b)
{
    vRealD ret;
    _for(i, 8, ret.v[i] = a.v[i] - b.v[i]);
    return ret;
}

// mul
static inline vRealF SimdMul(vRealF a, vRealF b)
{
    vRealF ret;
    _for(i, 16, ret.v[i] = a.v[i] * b.v[i]);
    return ret;
}
static inline vRealD SimdMul(vRealD a, vRealD b)
{
    vRealD ret;
    _for(i, 8, ret.v[i] = a.v[i] * b.v[i]);
    return ret;
}

// fmadd  dst = a*b+c
static inline vRealF SimdFmadd(vRealF a, vRealF b, vRealF c)
{
    vRealF ret;
    _for(i, 16, ret.v[i] = a.v[i] * b.v[i] + c.v[i]);
    return ret;
}
static inline vRealD SimdFmadd(vRealD a, vRealD b, vRealD c)
{
    vRealD ret;
    _for(i, 8, ret.v[i] = a.v[i] * b.v[i] + c.v[i]);
    return ret;
}

// fmsub  dst = a*b-c
static inline vRealF SimdFmsub(vRealF a, vRealF b, vRealF c)
{
    vRealF ret;
    _for(i, 16, ret.v[i] = a.v[i] * b.v[i] - c.v[i]);
    return ret;
}
static inline vRealD SimdFmsub(vRealD a, vRealD b, vRealD c)
{
    vRealD ret;
    _for(i, 8, ret.v[i] = a.v[i] * b.v[i] - c.v[i]);
    return ret;
}

#endif
