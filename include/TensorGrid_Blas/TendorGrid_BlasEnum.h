/**
 * @file TendorGrid_BlasEnum.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-19
 * 
 * @copyright Copyright (c) 2023
 * 
 */


enum TensorGrid_Precision { //
    TensorGridSingle   = 0b0001,
    TensorGridDouble   = 0b0010,
    TensorGridComplexF = 0b0101,
    TensorGridComplexD = 0b1010
};