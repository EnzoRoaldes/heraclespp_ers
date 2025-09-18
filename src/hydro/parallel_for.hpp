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

// template<int Nreg, class F>
// __global__ __maxnreg__(Nreg)
// void for_loop_3D(F const functor, const dim3 ib, const dim3 ie) {
//     // cache des builtins dans des registres (une seule fois)
//     const int bdx = blockDim.x, bdy = blockDim.y, bdz = blockDim.z;
//     const int gdx = gridDim.x,  gdy = gridDim.y,  gdz = gridDim.z;
//     const int tix = threadIdx.x, tiy = threadIdx.y, tiz = threadIdx.z;
//     const int bix = blockIdx.x, biy = blockIdx.y, biz = blockIdx.z;

//     // points de départ et strides (évite recomputes)
//     const int i0 = ib.x + bdx * bix + tix;
//     const int j0 = ib.y + bdy * biy + tiy;
//     const int k0 = ib.z + bdz * biz + tiz;
//     const int istep = bdx * gdx;
//     const int jstep = bdy * gdy;
//     const int kstep = bdz * gdz;

//     // boucles compactes (moins de variables vivantes)
//     for (int i = i0; i < ie.x; i += istep) {
//         for (int j = j0; j < ie.y; j += jstep) {
//             for (int k = k0; k < ie.z; k += kstep) {
//                 functor(i, j, k);
//             }
//         }
//     }
// }


template<int Nreg, class F>
void parallel_for_3D(std::array<int, 3> ib, std::array<int, 3> ie, F functor)
{
    dim3 ib_new(ib[0], ib[1], ib[2]);
    dim3 ie_new(ie[0], ie[1], ie[2]);
    // dim3 blocksPerGrid(19, 65, 129); // ancienne grille pour (256,128,256)
    dim3 blocksPerGrid(21, 161, 161); // nouvelle grille pour (320,320,320)
    dim3 threadsPerBlock(32, 2, 1);
    for_loop_3D<Nreg><<<blocksPerGrid, threadsPerBlock>>>(functor, ib_new, ie_new);
}