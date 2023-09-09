
BLAS_PATH= /usr/local

INCLUDE += -I${BLAS_PATH}/include
LINKS += -L${BLAS_PATH}/lib -Wl,-rpath,${BLAS_PATH}/lib -lopenblas

CC =gcc
CXX=g++
C_FLAGS  =
CXX_FLAGS= -O3

targets: tensorgrid_su3.bin

%.o: %.c
	${CC} ${C_FLAGS} ${INCLUDE} ${LINKS} -o $@ -c $<

%.o: %.cpp
	${CXX} ${CXX_FLAGS} ${INCLUDE} ${LINKS} -o $@ -c $<

tensorgrid_su3.bin: tensorgrid_su3.o
	${CXX} ${CXX_FLAGS} ${INCLUDE}  -o $@  $< ${LINKS}

clean:
	rm *.o *.bin -rf
