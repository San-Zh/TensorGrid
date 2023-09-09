#!/bin/bash
export OMP_NUM_THREADS=1

TENSOR="-DMAX_ROW=3 -DMAX_COL=3"
# PREC="-DPRECISION=SINGLE"

#gnu
# SIMDFLAGS="-ftree-vectorize"
SIMDFLAGS="-mavx"
SIMDFLAGS="-mavx512f" #-mavx512f" #-funroll-loops -march=native -ftree-loop-if-convert -DNORMAL_COMPLEX
# "-ftree-loop-if-convert" -march=skylake-avx512 corei7-avx

BLAS_PATH=/usr/local

INCLUDE=" -I${BLAS_PATH}/include"
LINKS=" -L${BLAS_PATH}/lib -Wl,-rpath,${BLAS_PATH}/lib -lopenblas"

LAYOUT="-DHAVE_BLAS"
# LAYOUT="-DAOS_LAYOUT -DHAVE_BLAS"

CXXFLAG+="${SIMDFLAGS} ${LAYOUT} -O3  ${INCLUDE}"

# CXXFLAG="-Ofast"

CXX=g++
# CXX=clang++

CFILE="tensorgrid_multiprec.cpp"

echo "${CXX} ${CXXFLAG} ${TENSOR} ${PREC}  ${CFILE}"

# 2^7= 4*4*4*2
SizeBase=32
echo "SizeBase: ${SizeBase}"

# Multiple=$(seq 1 1 64)
Multiple=(1 2 4 8 16 32 38 40 42 44 48 64 128 256 512 1024 2048 4096)
# gemv  16 pts  (3x3+3)xzgemv 12K  (3x3+3+3)xzgemv 15K
# cache     512/15        512/12
# Multiple=(34    36  38   42  44 46 )
# cache     1024/15        1024/12
# Multiple=(64 68 70 72   84 85 86 88 100 200 300)
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
    echo ""
done
