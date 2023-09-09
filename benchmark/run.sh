#!/bin/bash
export OMP_NUM_THREADS=1

TENSOR="-DMAX_ROW=3 -DMAX_COL=3"
# PREC="-DPRECISION=SINGLE"

# GSIZE='-DSIZE' -mfma -mavx512vl core-avx2   -march=skylake-avx512
# GCC8.2.0中关于向量化操作的选项有：
# -ftree-loop-vectorize、-ftree-slp-vectorize、
# -ftree-loop-if-convert、-ftree-vectorize、-fvect-cost-model=model、-fsimd-cost-model=model(cheap\dynamic\unlimited)。
# 前两个向量化选项默认情况下在-O3中已启用，这里不一一说明。

#gnu
# SIMDFLAGS="-mavx" #-ftree-vectorize" #-mavx512f" #-funroll-loops -march=native -ftree-loop-if-convert -DNORMAL_COMPLEX
SIMDFLAGS="-mavx512f" #-ftree-vectorize" #-mavx512f" #-funroll-loops -march=native -ftree-loop-if-convert -DNORMAL_COMPLEX
# "-ftree-loop-if-convert" -march=skylake-avx512 corei7-avx
#llvm clang
# SIMDFLAGS="-march=native -funroll-loops" # -Rpass-missed=loop-vectorize

# CXXFLAG+="-c -g -Wa,-adlhn " # 汇编+源码
BLAS_PATH=/usr/local

INCLUDE=" -I${BLAS_PATH}/include"
LINKS=" -L${BLAS_PATH}/lib -Wl,-rpath,${BLAS_PATH}/lib -lopenblas"

# LAYOUT="-DAOS_LAYOUT -DHAVE_BLAS"

CXXFLAG+="${SIMDFLAGS} ${LAYOUT} -O3  ${INCLUDE}"

# CXXFLAG="-Ofast"

CXX=g++
# CXX=clang++

CFILE="tensorgrid_su3.cpp"
# CFILE="tensorgrid_R3.cpp"
# CFILE="tensorgrid_cxypy.cpp"
# CFILE="tensorgrid_rxypy.cpp"
# CFILE="tensorgrid_aryio.cpp"

echo "${CXX} ${CXXFLAG} ${TENSOR} ${PREC}  ${CFILE}"

# 2^7= 4*4*4*2
SizeBase=16
echo "SizeBase: ${SizeBase}"

# Multiple=$(seq 1 1 64)
# Multiple=(1 2 4 8 16 32 38 40 42 44 48 64 128 256 512 1024 2048 4096)
# gemv  16 pts  (3x3+3)xzgemv 12K  (3x3+3+3)xzgemv 15K
# cache     512/15        512/12
# Multiple=(34    36  38   42  44 46 )
# cache     1024/15        1024/12
Multiple=(64 68 70 72   84 85 86 88 100 200 300)
# cache     17x1024/15        17x1024/12
# Multiple=(1150 1152  1100 1200 1200 )
# Multiple=$(seq 50 200 2100)
# Multiple=1
echo "Multiple: ${Multiple[*]}"

# for mul in ${Multiple[*]}; do
#     ${CXX} ${CXXFLAG} ${PREC} ${TENSOR} -DSIZE=$((${mul} * ${SizeBase})) ${CFILE} ${LINKS} -o a.out
#     ./a.out
# done

for mul in ${Multiple[*]}; do
    ${CXX} ${CXXFLAG} ${PREC} ${TENSOR} ${CFILE} -DSIZE=$((${mul} * ${SizeBase})) ${LINKS}
    for ((nr = 0; nr < 5; nr++)); do
        ./a.out
    done
    echo  ""
done
