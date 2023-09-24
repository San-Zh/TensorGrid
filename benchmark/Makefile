
SOFTROOT = ..

INCLUDE  = -I ${SOFTROOT}/include -I ${SOFTROOT}/benchmark/include
LDLIB=
# LINKS= -L./ -Wl,-rpath=./ 



CXX= clang++
CC = clang

CXX_FLAGS= -std=c++14 -O3
C_FLAGS  = -std=c99

SIMDFLAG = -mavx512f -mfma -fopenmp
PRESICION= -DPRECISION=SINGLE
TENSOR   = -DMROW=12 -DNCOL=12 -DKIN=12
TENSOR   = -DMROW=3 -DNCOL=4 -DKIN=3
SIZE     = -DBASEGRID=16


HAVE_BLAS = YES
BLAS_PATH =/usr/local

ifeq (${HAVE_BLAS},YES)
CXX_FLAGS += -DHAVE_BLAS
INCLUDE  += -I${BLAS_PATH}/include
LDLIB += -L${BLAS_PATH}/lib/tmp -Wl,-rpath,${BLAS_PATH}/lib/tmp 
LINKS += -lopenblas
endif

CXX_FLAGS += ${PRESICION} ${SIZE} ${TENSOR} ${SIMDFLAG} ${INCLUDE}
C_FLAGS += ${PRESICION} ${SIZE} ${TENSOR} ${SIMDFLAG} ${INCLUDE}

# targets = Benchmark_Gemv_MxN.bin Benchmark_Gemv_Check.bin

all: ${targets}

%.o: %.cpp
	${CXX} ${CXX_FLAGS} -o $@ -c $<

%.o: %.c
	${CC} ${C_FLAGS} -o $@ -c $<


%.bin: %.o
	${CXX} ${CXX_FLAGS} ${INCLUDE}  -o $@  $< ${LDLIB} ${LINKS}


############################

Benchmark_ComplexGemv.bin: Benchmark_ComplexGemv.cpp
	${CXX} ${CXX_FLAGS} -o $@  $< ${LDLIB} ${LINKS}

Benchmark_ComplexGemv_Check.bin: Benchmark_ComplexGemv.cpp
	${CXX} ${CXX_FLAGS} -DCHECK -o $@  $< ${LDLIB} ${LINKS}


Benchmark_ComplexGemm.bin: Benchmark_ComplexGemm.cpp
	${CXX} ${CXX_FLAGS} -o $@  $< ${LDLIB} ${LINKS}

Benchmark_ComplexGemm_Check.bin: Benchmark_ComplexGemm.cpp
	${CXX} ${CXX_FLAGS} -DCHECK -o $@  $< ${LDLIB} ${LINKS}


Benchmark_RealGemm.bin: Benchmark_RealGemm.cpp
	${CXX} ${CXX_FLAGS} -o $@  $< ${LDLIB} ${LINKS}

Benchmark_RealGemm_Check.bin: Benchmark_RealGemm.cpp
	${CXX} ${CXX_FLAGS} -DCHECK -o $@  $< ${LDLIB} ${LINKS}


####################

real_gemm:
	make clean
	make Benchmark_RealGemm.bin
	export OMP_NUM_THREADS=1
	./Benchmark_RealGemm.bin 

real_gemmcheck:
	make clean
	make Benchmark_RealGemm_Check.bin
	export OMP_NUM_THREADS=1
	./Benchmark_RealGemm_Check.bin 


complex_gemm:
	make clean
	make Benchmark_ComplexGemm.bin
	export OMP_NUM_THREADS=1
	./Benchmark_ComplexGemm.bin 

complex_gemmcheck:
	make clean
	make Benchmark_ComplexGemm_Check.bin
	export OMP_NUM_THREADS=1
	./Benchmark_ComplexGemm_Check.bin 


complex_gemv:
	make clean
	make Benchmark_ComplexGemv.bin
	export OMP_NUM_THREADS=1
	./Benchmark_ComplexGemv.bin 

complex_gemvcheck: 
	make clean
	make Benchmark_ComplexGemv_Check.bin
	export OMP_NUM_THREADS=1
	./Benchmark_ComplexGemv_Check.bin


clean: 
	rm -rf *.o *.bin



