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

enum Enum_TensorGridPrecision_t {
    REALF    = 0b0001,
    REALD    = 0b0010,
    COMPLEXF = 0b0101,
    COMPLEXD = 0b1010
};



/**
 * @brief
 *
 */
enum EnumBit_4DGridEvenOdd_t {
    EEEO = 0b0001, //  1  O
    EEOE = 0b0010, //  2  O
    EOEE = 0b0100, //  4  O
    OEEE = 0b1000, //  8  O
    EOOO = 0b0111, //  7  O
    OEOO = 0b1011, // 11  O
    OOEO = 0b1101, // 13  O
    OOOE = 0b1110, // 14  O

    EEEE = 0b0000, //  0  E
    EEOO = 0b0011, //  3  E
    EOEO = 0b0101, //  5  E
    EOOE = 0b0110, //  6  E
    OEEO = 0b1001, //  9  E
    OEOE = 0b1010, // 10  E
    OOEE = 0b1100, // 12  E
    OOOO = 0b1111  // 15  E
};


enum EnumBit_4DGridDir_t {
    T_DIR = OEEE,
    Z_DIR = EOEE,
    Y_DIR = EEOE,
    X_DIR = EEEO // x dir
};


/**
 * @brief 
 * 
 * @tparam GridEO 
 * @tparam DirEnum 
 */
template <EnumBit_4DGridEvenOdd_t GridEO, EnumBit_4DGridDir_t DirEnum>
struct EvenOddNeighborGrid {
    // enum { ID = (GridEO & DirEnum) ? (GridEO - DirEnum) : (GridEO + DirEnum) };
    // enum { ID = (GridEO | (GridEO & DirEnum)) };
};


//////////////////////////////////////////

// template <EnumBit_4DGridEvenOdd_t GridEO, EnumBit_4DGridDir_t DirEnum>
// EnumBit_4DGridEvenOdd_t NeibGrid()
// {
//     return static_cast<EnumBit_4DGridEvenOdd_t>((GridEO & DirEnum) ? (GridEO - DirEnum)
//                                                                    : (GridEO + DirEnum));
// }

EnumBit_4DGridEvenOdd_t NeibGrid(const EnumBit_4DGridEvenOdd_t &GridEO,
                                 const EnumBit_4DGridDir_t     &DirEnum)
{
    return static_cast<EnumBit_4DGridEvenOdd_t>((GridEO & DirEnum) ? (GridEO - DirEnum)
                                                                   : (GridEO + DirEnum));
}



// template <EnumBit_4DGridEvenOdd_t GridEO, EnumBit_4DGridDir_t DirEnum>
// constexpr EnumBit_4DGridEvenOdd_t
//     NeibGrid = static_cast<EnumBit_4DGridEvenOdd_t>((GridEO & DirEnum) ? (GridEO - DirEnum)
//                                                                        : (GridEO + DirEnum));

// template <EnumBit_4DGridEvenOdd_t GridEO, EnumBit_4DGridDir_t DirEnum>
// constexpr EnumBit_4DGridEvenOdd_t
//     NeibGrid = static_cast<EnumBit_4DGridEvenOdd_t>(GridEO - (GridEO & DirEnum));



void EnumBit_Print(const EnumBit_4DGridEvenOdd_t &GridEO)
{
    printf("Enum %d named %c%c%c%c\n", GridEO, (GridEO & T_DIR) ? 'O' : 'E',
           (GridEO & Z_DIR) ? 'O' : 'E', (GridEO & Y_DIR) ? 'O' : 'E',
           (GridEO & X_DIR) ? 'O' : 'E');
}


void EnumBitString(char *s, const EnumBit_4DGridEvenOdd_t &GridEO)
{
    sprintf(s, "%c%c%c%c", (GridEO & T_DIR) ? '1' : '0', (GridEO & Z_DIR) ? '1' : '0',
            (GridEO & Y_DIR) ? '1' : '0', (GridEO & X_DIR) ? '1' : '0');
}

void EnumBitName(char *s, const EnumBit_4DGridEvenOdd_t &GridEO)
{
    sprintf(s, "%c%c%c%c", (GridEO & T_DIR) ? 'O' : 'E', (GridEO & Z_DIR) ? 'O' : 'E',
            (GridEO & Y_DIR) ? 'O' : 'E', (GridEO & X_DIR) ? 'O' : 'E');
}
