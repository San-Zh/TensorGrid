#!/bin/bash
export OMP_NUM_THREADS=1

TENSOR="-DMAX_ROW=3 -DMAX_COL=3"

#gnu
SIMDFLAGS="-ftree-vectorize" 
SIMDFLAGS="-mavx -ftree-vectorize" 
SIMDFLAGS="-mavx512f -ftree-vectorize" #-mavx512f" #-funroll-loops -march=native -ftree-loop-if-convert -DNORMAL_COMPLEX
# "-ftree-loop-if-convert" -march=skylake-avx512 corei7-avx

# CXXFLAG+="${SIMDFLAGS} -O0"

CXXFLAG="-O3"

CXX=g++
# CXX=clang++

CFILE="tensorgrid_multiprec.cpp"

echo "${CXX} ${CXXFLAG} ${TENSOR} ${PREC}  ${CFILE}"

# Multiple=$(seq 1 1 64)
Multiple=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
# Multiple=(2048)
# Multiple=(16)
echo "Multiple: ${Multiple[*]}"

# 2^7= 4*4*4*2
SizeBase=512
echo "SizeBase: ${SizeBase}"

for mul in ${Multiple[*]}; do
    ${CXX} ${CXXFLAG} ${PREC} ${TENSOR} ${CFILE} -DSIZE=$((${mul} * ${SizeBase}))
    ./a.out
done