
SOFTROOT = ..

INCLUDE  = -I ${SOFTROOT}/include -I ${SOFTROOT}/benchmark/include
LDLIB=
LINKS= -L./ -Wl,-rpath=./ 



CXX= g++
CC = gcc

CXX_FLAGS= -std=c++14 -O3 #-DDEBUG
C_FLAGS  = -std=c99

SIMDFLAG = -mavx512f -mfma
PRESICION= -DPRECISION=SINGLE
TENSOR   = -DMAX_ROW=2 -DMAX_COL=4
SIZE     = -DBASEGRID=512


HAVE_BLAS = NO
BLAS_PATH =/usr/local

ifeq (${HAVE_BLAS},YES)
INCLUDE  += -I${BLAS_PATH}/include
LDLIB+= -L${BLAS_PATH}/lib -Wl,-rpath,${BLAS_PATH}/lib 
LINKS += -lopenblas
endif

CXX_FLAGS += ${PRESICION} ${SIZE} ${TENSOR} ${SIMDFLAG} ${INCLUDE}
C_FLAGS += ${PRESICION} ${SIZE} ${TENSOR} ${SIMDFLAG} ${INCLUDE}

targets = Benchmark_SU3_gemv.bin Benchmark_Check.bin su3.bin

all: ${targets}

%.o: %.cpp
	${CXX} ${CXX_FLAGS} -o $@ -c $<

%.o: %.c
	${CC} ${C_FLAGS} -o $@ -c $<


%.bin: %.o
	${CXX} ${CXX_FLAGS}  -o $@  $< ${LDLIB} ${LINKS}


# Benchmark_SU3_gemv.bin: Benchmark_SU3_gemv.o
# 	${CXX} ${CXX_FLAGS}  -o $@  $< ${LDLIB} ${LINKS}

Benchmark_Check.bin: Benchmark_Check.o
	${CXX} ${CXX_FLAGS}  -o $@  $< ${LDLIB} ${LINKS}


su3: su3.bin
	./su3.bin

check: 
	make clean
	make Benchmark_Check.bin
	./Benchmark_Check.bin

run: Benchmark_SU3_gemv.bin
	./Benchmark_SU3_gemv.bin 
	make clean

clean: 
	rm -rf *.o ${targets} *.bin



