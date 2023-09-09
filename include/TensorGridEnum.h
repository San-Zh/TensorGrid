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

enum EnumBit_4DGridDir_t {
  X_DIR = 0b0001,
  Y_DIR = 0b0010,
  Z_DIR = 0b0100,
  T_DIR = 0b1000
};

/**
 * @brief
 *
 */
enum EnumBit_4DGridEvenOdd_t {
  EEEO = 0b0001,  //  1  O
  EEOE = 0b0010,  //  2  O
  EOEE = 0b0100,  //  4  O
  OEEE = 0b1000,  //  8  O
  EOOO = 0b0111,  //  7  O
  OEOO = 0b1011,  // 11  O
  OOEO = 0b1101,  // 13  O
  OOOE = 0b1110,  // 14  O

  EEEE = 0b0000,  //  0  E
  EEOO = 0b0011,  //  3  E
  EOEO = 0b0101,  //  5  E
  EOOE = 0b0110,  //  6  E
  OEEO = 0b1001,  //  9  E
  OEOE = 0b1010,  // 10  E
  OOEE = 0b1100,  // 12  E
  OOOO = 0b1111   // 15  E
};

/**
 * @brief 
 * 
 * 
 * @tparam GridEO 
 * @tparam DirEnum 
 */
template <EnumBit_4DGridEvenOdd_t GridEO, EnumBit_4DGridDir_t DirEnum>
struct EvenOddNeighborGrid {
  enum { ID = (GridEO & DirEnum) ? (GridEO - DirEnum) : (GridEO + DirEnum) };
};
