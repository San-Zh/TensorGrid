/**
 * @file Test_Geometry.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-05-12
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <iostream>

#include "Grid.h"

using namespace std;

int main(int argc, char **argv) {
  size_t G[2] = {11, 13};

  size_t T[3] = {2, 3, 4};

  size_t g[4] = {4, 5, 6, 7};

  Grid<double, 4> GA(g);


  return 0;
}
