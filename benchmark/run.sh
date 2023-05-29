#!/bin/bash
export OMP_NUM_THREADS=1

TENSOR="-DMAX_ROW=3 -DMAX_COL=3"
PREC="-DPRECISION=SINGLE"

# GSIZE='-DSIZE' -mfma -mavx512vl core-avx2   -march=skylake-avx512
# GCC8.2.0中关于向量化操作的选项有：
# -ftree-loop-vectorize、-ftree-slp-vectorize、
# -ftree-loop-if-convert、-ftree-vectorize、-fvect-cost-model=model、-fsimd-cost-model=model(cheap\dynamic\unlimited)。
# 前两个向量化选项默认情况下在-O3中已启用，这里不一一说明。

#gnu
# SIMDFLAGS="-mfma -ftree-vectorize -march=native " # -march=native
# "-ftree-loop-if-convert" -march=skylake-avx512 corei7-avx
#llvm clang
SIMDFLAGS="-march=native -funroll-loops" # -Rpass-missed=loop-vectorize

# CXXFLAG+="-c -g -Wa,-adlhn " # 汇编+源码
CXXFLAG+="-O3 ${SIMDFLAGS}"
CXX=g++
# CXX=clang++

CFILE="tensorgrid_su3.cpp"
# CFILE="tensorgrid_R3.cpp"

echo "${CXX} ${CXXFLAG} ${TENSOR} ${PREC}  ${CFILE}"

# Multiple=$(seq 1 1 64)
Multiple=( 1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
# Multiple=(2048)
# Multiple=(2)
echo "Multiple: ${Multiple[*]}"

# 2^7= 4*4*4*2
SizeBase=512
echo "SizeBase: ${SizeBase}"

for mul in ${Multiple[*]};do
    ${CXX} ${CXXFLAG} ${PREC} ${TENSOR} ${CFILE} -DSIZE=$[${mul}*${SizeBase}]
    ./a.out
done