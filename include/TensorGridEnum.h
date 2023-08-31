/**
 * @file TensorGridEnum.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-08-31
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

enum Enum_4DGridDir_t {
  X_DIR = 0b0001,
  Y_DIR = 0b0010,
  Z_DIR = 0b0100,
  T_DIR = 0b1000
};

/**
 * @brief
 *
 */
enum Enum_4DGridEvenOdd_t {
  EEEE = 0b0000,  //  0  E
  EEEO = 0b0001,  //  1  O
  EEOE = 0b0010,  //  2  O
  EEOO = 0b0011,  //  3  E
  EOEE = 0b0100,  //  4  O
  EOEO = 0b0101,  //  5  E
  EOOE = 0b0110,  //  6  E
  EOOO = 0b0111,  //  7  O
  OEEE = 0b1000,  //  8  O
  OEEO = 0b1001,  //  9  E
  OEOE = 0b1010,  // 10  E
  OEOO = 0b1011,  // 11  O
  OOEE = 0b1100,  // 12  E
  OOEO = 0b1101,  // 13  O
  OOOE = 0b1110,  // 14  O
  OOOO = 0b1111   // 15  E
};

template <Enum_4DGridEvenOdd_t DEO, Enum_4DGridDir_t DirEnum>
class TensorGridEnum {
  enum { Neighbor = (DEO | DirEnum) ? (DEO - DirEnum) : (DEO - DirEnum) };
};
