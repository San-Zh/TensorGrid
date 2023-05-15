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
#include "Geometry.h"

using namespace std;

int main(int argc, char **argv)
{
    size_t G[2] = {11, 13};

    size_t T[3] = {2, 3, 4};

    cout << "size_t G[2] {" << G[0] << ", " << G[1] << "} G[2]=" << G[2] << "  &G[2]" << &G[2]
         << endl;
    cout << "size_t T[3] {" << T[0] << ", " << T[1] << ", " << T[2] << "}  &T[0]" << &T[0] << "  T "
         << T << "\n"
         << endl;

    Geometry<2> A(G); // Geometry(const DataVect &_vec)
    cout << "A " << A << "  volume= " << A.volume() << " dim=" << A.numAxes << endl;

    Geometry<3> B({12, 14, 16}); // Geometry(const DataVect &_vec)
    cout << "B " << B << "  volume= " << B.volume() << " dim=" << B.numAxes << endl;

    Geometry<2> C(10); // Geometry(const DataType &_a)
    cout << "C " << C << "  volume= " << C.volume() << " dim=" << C.numAxes << endl;

    Geometry<3> D(B); // Geometry(const Geometry<LEN> &_vec)
    cout << "D(B)      " << D << "  volume= " << D.volume() << " dim=" << D.numAxes << endl;

    // Geometry<3> E(A); // no matching function for call to ‘Geometry<3>::Geometry(Geometry<2>&)
    // Geometry<3> E(G); // error: invalid conversion from ‘std::size_t*’ {aka ‘long unsigned int*’} to ‘Geometry<3>::DataType’ {aka ‘long unsigned int’} [-fpermissive]
    Geometry<3> E(T);
    cout << "E(T)      " << E << "  volume= " << E.volume() << " dim=" << E.numAxes << endl;

    Geometry<3> &F = D;
    cout << "\nF(G)      " << F << "  volume= " << F.volume() << " dim=" << F.numAxes << endl;

    B[0] = A[0];
    B[1] = A[1];
    B[2] = A[2];
    cout << "B[] = A[] " << B << "  volume= " << B.volume() << " dim=" << B.numAxes << "  &A[2]"
         << &A[2] << endl;
    cout << "B = D     " << (B = D) << "  volume= " << B.volume() << " dim=" << B.numAxes << endl;

    // Geometry<3> B([1, 2, 3]);

    // test<int, Geometry<2>> TA(3, A);

    return 0;
}
