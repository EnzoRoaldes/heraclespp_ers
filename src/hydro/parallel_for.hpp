#pragma once

#include <Kokkos_Core.hpp>
#include <string>

// 3D loop utility for Idefix-style reconstruction
template <typename Function>
inline void idefix_for(const std::string & NAME,
                       const int & KB, const int & KE,
                       const int & JB, const int & JE,
                       const int & IB, const int & IE,
                       Function function) {
    const int NK = KE - KB;
    const int NJ = JE - JB;
    const int NI = IE - IB;
    const int NKNJNI = NK * NJ * NI;
    const int NJNI = NJ * NI;

    Kokkos::parallel_for(NAME, NKNJNI,
        KOKKOS_LAMBDA (const int& IDX) {
            int k = IDX / NJNI;
            int j = (IDX - k * NJNI) / NI;
            int i = IDX - k * NJNI - j * NI;
            k += KB;
            j += JB;
            i += IB;
            function(i, j, k);
        }
    );
}


// template <typename Function>
// inline void idefix_for(const std::string & NAME,
//                        const int & KB, const int & KE,
//                        const int & JB, const int & JE,
//                        const int & IB, const int & IE,
//                        const std::array<int,3> & tiling,
//                        Function function) {
//     const int NK = KE - KB;
//     const int NJ = JE - JB;
//     const int NI = IE - IB;
//     const int NKNJNI = NK * NJ * NI;
//     const int NJNI = NJ * NI;

//     Kokkos::parallel_for(NAME, NKNJNI, tiling,
//         KOKKOS_LAMBDA (const int& IDX) {
//             int k = IDX / NJNI;
//             int j = (IDX - k * NJNI) / NI;
//             int i = IDX - k * NJNI - j * NI;
//             k += KB;
//             j += JB;
//             i += IB;
//             function(i, j, k);
//         }
//     );
// }



// F : la fonction a tester (ici face_reconstruction)
// Nreg : le nombre de registres max
template<int Nreg, class F>
__global__ __maxnreg__(Nreg) void for_loop_3D(F const functor, const dim3 ib, const dim3 ie) {
    for(int i =ib.x + blockDim.x * blockIdx.x + threadIdx.x; i < ie.x; i+= blockDim.x * gridDim.x)
    {
        for(int j =ib.y + blockDim.y * blockIdx.y + threadIdx.y; j < ie.y; j+= blockDim.y * gridDim.y)
        {
            for(int k =ib.z + blockDim.z * blockIdx.z + threadIdx.z; k < ie.z; k+= blockDim.z * gridDim.z)
            {
                functor(i, j, k);
            }
        }
    }
}

template<int Nreg, class F>
void parallel_for_3D(std::array<int, 3> ib, std::array<int, 3> ie, F functor)
{
    dim3 ib_new(ib[0], ib[1], ib[2]);
    dim3 ie_new(ie[0], ie[1], ie[2]);
    dim3 blocksPerGrid(19, 65, 129);
    dim3 threadsPerBlock(32, 2, 1);
    for_loop_3D<Nreg><<<blocksPerGrid, threadsPerBlock>>>(functor, ib_new, ie_new);
}