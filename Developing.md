
# Developing

[TOC]

## 0. About Simd Vector Uniformed Struct

## 1.About Simd Tuple Types and Kernals
###  *iVector*
###  *iMaxtrix*



## 1. About TensorGrid Blas

### *TensorGrid_zgemv*
在 *<u>inlude/TensorGrid_blas.h</u>* 中有**三个**版本,分别为

```c++
// v0.0.0
template <unsigned M, unsigned N, typename Tp>
void TensorGrid_complex_gemv(const Tp *A, const Tp *X, Tp *Y, const size_t gridSize);
{
    vComplex<Tp> Vd[M];
    vComplex<Tp> Mn[M];
    vComplex<Tp> vs; 
    ........     Vd[m] = SimdFmadd(Mn[m], vs, Vd[m]); .........
        for (int m = 0; m < M; m++) { SimdStore(&Vdp[m][0][v], &Vdp[m][1][v], Vd[m]); }
}

// v0.0.1
template <unsigned M, unsigned N, typename Tp>
void TensorGrid_complex_gemv_v1(const Tp *A, const Tp *X, Tp *Y, const size_t gridSize)
{......    kernal_simd_aXpY(vs, Vm, Vd);   ........ }

// v0.0.2
template <unsigned M, unsigned N, typename Tp>
void TensorGrid_complex_gemv_v2(const Tp *A, const Tp *X, Tp *Y, const size_t gridSize)
{......    kernal_simd_XdotY(vdes, Vm, Vs); ........ }

```

- v0.0.0:   
  仅使用了Simd vtypes 构建数组，如 `vComplex<Tp> Vd[N]`, 未使用SimdTurpleTypes 以及对应的kernal  

- v0.0.1:  
  使用**外积**方式计算，使用了 *`iVertor`*,计算过程任然使用循环+封装的Simd运算，如`SimdLoad(),SimdFmadd()`,但并未使用与之对应的kernal_simd_xxx (*include/Simd/SimdTupleKernals.h*)；  
  

- v0.0.2
  使用**外积**方式计算，使用了 *`iVertor`*,但并未使用与之对应的kernal_simd_xxx (*include/Simd/SimdTupleKernals.h*)


## 3. About Benchmark

### 3.1. About Basic Style: Generic Implementation  
#### 3.1.1 

#### 3.1.2 


## NEWS & TODO
- ~~TODO 使 Gemv API 形式与 Gemm 相似, (2023.09.23)~~  
  ----------- Done, (2023.09.24)
- TO use `loadu()/storeu()`, instead of original ``load()/store()``; otherwise, it would cause ***SegmentFault***  when performing **shift operation** because not as the **aligned** memory requirements. (2023.09.24)  
