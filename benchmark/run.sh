export OMP_NUM_THREADS=1
# MSIZE=4
# TENSOR='-DCOLUMN='$MSIZE' -DROW='$MSIZE
TENSOR='-DMAX_ROW=3 -DMAX_COL=3'
PREC='-DPRECISION=SINGLE'
# GSIZE='-DSIZE' -mfma -mavx512vl 
CXXFLAG='-O3 -fopenmp'

# CFILE="tensorgrid_R3.cpp"
CFILE="tensorgrid_su3.cpp"

echo "cxx ${CXXFLAG} ${TENSOR} ${PREC}  ${CFILE}" 

# g++ $CXXFLAG $PREC $TENSOR ${CFILE} -DSIZE=8
# ./a.out
# g++ $CXXFLAG $PREC $TENSOR ${CFILE} -DSIZE=16
# ./a.out
# g++ $CXXFLAG $PREC $TENSOR ${CFILE} -DSIZE=32
# ./a.out
# g++ $CXXFLAG $PREC $TENSOR ${CFILE} -DSIZE=64
# ./a.out
g++ $CXXFLAG $PREC $TENSOR ${CFILE} -DSIZE=128
./a.out
g++ $CXXFLAG $PREC $TENSOR ${CFILE} -DSIZE=256
./a.out
g++ $CXXFLAG $PREC $TENSOR ${CFILE} -DSIZE=512
./a.out
g++ $CXXFLAG $PREC $TENSOR ${CFILE} -DSIZE=1024       # 256   64
./a.out
g++ $CXXFLAG $PREC $TENSOR ${CFILE} -DSIZE=1024*2     # 128   32
./a.out
g++ $CXXFLAG $PREC $TENSOR ${CFILE} -DSIZE=1024*4     # 64    16
./a.out
g++ $CXXFLAG $PREC $TENSOR ${CFILE} -DSIZE=1024*8     # 32    8
./a.out
g++ $CXXFLAG $PREC $TENSOR ${CFILE} -DSIZE=1024*16    # 16    4
./a.out
g++ $CXXFLAG $PREC $TENSOR ${CFILE} -DSIZE=1024*32    # 8     2
./a.out
g++ $CXXFLAG $PREC $TENSOR ${CFILE} -DSIZE=1024*64    # 4     1
./a.out
