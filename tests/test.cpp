#include <cstdio>

#include "TensorGridEnum.h"



void EnumBit_NeiborInfo(const EnumBit_4DGridEvenOdd_t &GridEO)
{
    char s[5];
    char bs[5];
    EnumBitName(s, GridEO);
    EnumBitString(bs, GridEO);
    printf("%s(0b%s) 's Neighbors:\n", s, bs);
    EnumBitName(s, NeibGrid(GridEO, X_DIR));
    EnumBitString(bs, NeibGrid(GridEO, X_DIR));
    printf("  %s (0b%s)\n", s, bs);
    EnumBitName(s, NeibGrid(GridEO, Y_DIR));
    EnumBitString(bs, NeibGrid(GridEO, Y_DIR));
    printf("  %s (0b%s)\n", s, bs);
    EnumBitName(s, NeibGrid(GridEO, Z_DIR));
    EnumBitString(bs, NeibGrid(GridEO, Z_DIR));
    printf("  %s (0b%s)\n", s, bs);
    EnumBitName(s, NeibGrid(GridEO, T_DIR));
    EnumBitString(bs, NeibGrid(GridEO, T_DIR));
    printf("  %s (0b%s)\n", s, bs);
}


int main()
{
    //   constexpr EnumBit_4DGridDir_t d = X_DIR;
    //   constexpr EnumBit_4DGridEvenOdd_t GEO = EEOO;


    // char * EOBitString = EnumBitToString<EOEO>();

    // constexpr EnumBit_4DGridEvenOdd_t Neib  = EvenOddNeighborGrid<EO_ID, Z_DIR>::ID;

    for (size_t i = 0; i < 16; i++) {
        EnumBit_4DGridEvenOdd_t EO_ID = static_cast<EnumBit_4DGridEvenOdd_t>(i);

        char cs[5], bs[5];
        EnumBitName(cs, EO_ID);
        EnumBitString(bs, EO_ID);
        printf(" %s = %s \n", cs, bs);

        // EnumBit_NeiborInfo(EO_ID);
        // EnumBit_Print(EvenOddNeighborGrid<EO_ID, T_DIR>::ID);
        // EnumBit_Print(EvenOddNeighborGrid<EO_ID, T_DIR>::ID);

        // printf("EOEO = %c \n", EnumBitChar_Dir<EOEO, X_DIR>::Char);
        // printf("EOEO = %c \n", EnumBitToChar_1bit<EOEO, Y_DIR>());
    }
    return 0;
}