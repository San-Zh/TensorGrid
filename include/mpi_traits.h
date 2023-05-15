#pragma once

#include <mpi.h>

/**
 * @brief 
 * 
 * @tparam Tp 
 */
template <typename Tp>
class MpiTrait;

template <>
class MpiTrait<int> {
  public:
    const MPI_Datatype Type = MPI_INT;
};

template <>
class MpiTrait<double> {
  public:
    const MPI_Datatype Type = MPI_DOUBLE;
};

template <>
class MpiTrait<float> {
  public:
    const MPI_Datatype Type = MPI_FLOAT;
};

/**
 * @brief 
 * 
 * @tparam Tp 
 * @return MPI_Datatype 
 */
template <typename Tp>
MPI_Datatype MPIDataType();
template <>
MPI_Datatype MPIDataType<int>()
{
    return MPI_INT;
}
template <>
MPI_Datatype MPIDataType<double>()
{
    return MPI_DOUBLE;
}
template <>
MPI_Datatype MPIDataType<float>()
{
    return MPI_FLOAT;
}