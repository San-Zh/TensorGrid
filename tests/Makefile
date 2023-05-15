CC = gcc
CXX = g++

INCLUDE := ../include
SOURCE := ./

VPATH = $(SOURCE):${INCLUDE}

targets: Test_Geometry.bin Test_Grid.bin

.SUFFIXES: .cpp .c
.cpp.o:
	${CXX} ${CXX_FLAGS} -o $@ -c $< -I${INCLUDE}

.c.o:
	${CC} ${C_FLAGS} -o $@ -c $< -I${INCLUDE}

Test_Geometry.bin: Test_Geometry.o
	${CXX} ${CXX_FLAGS} -o $@  $< -I${INCLUDE}

Test_Grid.bin: Test_Grid.o
	${CXX} ${CXX_FLAGS} -o $@  $< -I${INCLUDE}


.PHONY: targets clean fmt
fmt:
	@#cd . && git ls-files -- '*.cpp' '*.h' | xargs clang-format -i -style=file
	find . -name "*.cpp" -or -name "*.h" | xargs clang-format -i -style=file

clean:
	rm *.o *.bin -rf