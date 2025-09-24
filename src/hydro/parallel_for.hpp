#pragma once

#include <Kokkos_Core.hpp>
#include <string>

inline dim3 g_threads_override{0,0,0};
inline dim3 g_blocks_override{0,0,0};
inline bool g_use_override = false;
inline void set_cuda_launch(dim3 threads, dim3 blocks) {
    g_threads_override = threads;
    g_blocks_override  = blocks;
    g_use_override = true;
}
inline void reset_cuda_launch() {
    g_threads_override = dim3{0,0,0};
    g_blocks_override  = dim3{0,0,0};
    g_use_override = false;
}

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

// F : la fonction a tester (ici face_reconstruction)
// Nreg : le nombre de registres max
template<int Nreg, class F>
__global__ __maxnreg__(Nreg) void for_loop_3D(F const functor, const dim3 ib, const dim3 ie) 
{
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
void parallel_for_3D(std::array<int, 3> ib, std::array<int, 3> ie, F functor, dim3 threadsPerBlock = dim3(64,2,1), dim3 blocksPerGrid = dim3(0,0,0))
{
    if (g_use_override){
        threadsPerBlock = g_threads_override;
        blocksPerGrid = g_blocks_override;
    }

    const int nx = ie[0]-ib[0], ny = ie[1]-ib[1], nz = ie[2]-ib[2];

    if (threadsPerBlock.x==0) {
        threadsPerBlock.x = 64;
        threadsPerBlock.y = 2;
        threadsPerBlock.z = 1;
    }
    if (blocksPerGrid.x==0) {
        blocksPerGrid.x  = (nx + threadsPerBlock.x - 1) / threadsPerBlock.x;
        blocksPerGrid.y  = (ny + threadsPerBlock.y - 1) / threadsPerBlock.y;
        blocksPerGrid.z  = (nz + threadsPerBlock.z - 1) / threadsPerBlock.z;
    }

    dim3 ib_new(ib[0], ib[1], ib[2]);
    dim3 ie_new(ie[0], ie[1], ie[2]);
    for_loop_3D<Nreg><<<blocksPerGrid, threadsPerBlock>>>(functor, ib_new, ie_new);
}


template<int Nreg, class F>
void parallel_for_3D_v2(std::array<int, 3> ib, std::array<int, 3> ie, F functor, dim3 threadsPerBlock = dim3(64,2,1), dim3 blocksPerGrid = dim3(0,0,0))
{
    if (g_use_override){
        threadsPerBlock = g_threads_override;
        blocksPerGrid = g_blocks_override;
    }

    const int nx = ie[0]-ib[0], ny = ie[1]-ib[1], nz = ie[2]-ib[2];

    if (threadsPerBlock.x==0) {
        threadsPerBlock = {64,2,1};
    }

    if (blocksPerGrid.x==0) {
        // 132 SMs on H100 and we aim for 2 blocks per SM ie 264 blocks total
        blocksPerGrid = {4, 6, 11};
    }

    dim3 ib_new(ib[0], ib[1], ib[2]);
    dim3 ie_new(ie[0], ie[1], ie[2]);
    for_loop_3D<Nreg><<<blocksPerGrid, threadsPerBlock>>>(functor, ib_new, ie_new);
}