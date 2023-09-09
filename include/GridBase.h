/**
 * @file GridBase.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-09-05
 *
 * @copyright Copyright (c) 2023
 *
 */

template <typename TF>
struct GridBase {
  TF* G;
};

template <typename TF, unsigned int NumDim>
struct GridBase_EvenOdd {
  GridBase<TF> G[1 << NumDim];
};
