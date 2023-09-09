#include <cstdio>

#include "TensorGridEnum.h"

int main() {
  //   constexpr EnumBit_4DGridDir_t d = X_DIR;
  //   constexpr EnumBit_4DGridEvenOdd_t GEO = EEOO;

  int a = EvenOddNeighborGrid<OOOO, Z_DIR>::ID;
  constexpr int b = (OEOE | Z_DIR);

  printf("b = %d \n",b);

  return 0;
}