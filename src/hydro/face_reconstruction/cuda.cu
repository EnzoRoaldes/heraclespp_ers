// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file face_reconstruction.hpp
//!

#pragma once

#include <cassert>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <grid.hpp>
#include <kokkos_shortcut.hpp>
#include <kronecker.hpp>
#include <ndim.hpp>
#include <range.hpp>
#include <parallel_for.hpp>

#include "../face_reconstruction.hpp"
#include "slope_limiters.hpp"

namespace novapp
{

template <typename SlopeLimiter>
class FaceReconstructionCuda : public IFaceReconstruction
{

private:
    SlopeLimiter m_slope_limiter;
    bool m_enable_timer;

public:
    explicit FaceReconstructionCuda(SlopeLimiter limiter, bool enable_timer = false)
        : m_slope_limiter(limiter), m_enable_timer(enable_timer) {}

    void execute(
        Range const& range,
        Grid const& grid,
        KV_cdouble_3d const& var,
        KV_double_5d const& var_rec) const override
    {
        assert(equal_extents({0, 1, 2}, var, var_rec));
        assert(var_rec.extent(3) == 2);
        assert(var_rec.extent(4) == ndim);
        
        KV_cdouble_1d const dx = grid.dx;
        KV_cdouble_1d const dy = grid.dy;
        KV_cdouble_1d const dz = grid.dz;
        
        auto const& slope_limiter = m_slope_limiter;

        if (m_enable_timer) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            const auto [begin, end] = cell_range_std(range);
            int Nx = end[0] - begin[0];
            int Ny = end[1] - begin[1];
            int Nz = end[2] - begin[2];

            parallel_for_3D<50>(begin, end, [=] __device__ __host__ (int i, int j, int k){
                for (int idim = 0; idim < ndim; ++idim)
                {
                    auto const [i_m, j_m, k_m] = lindex(idim, i, j, k); // i - 1
                    auto const [i_p, j_p, k_p] = rindex(idim, i, j, k); // i + 1
                    double const dl   = kron(idim,0) * dx(i)
                                    + kron(idim,1) * dy(j)
                                    + kron(idim,2) * dz(k);
                    double const dl_m = kron(idim,0) * dx(i_m)
                                    + kron(idim,1) * dy(j_m)
                                    + kron(idim,2) * dz(k_m);
                    double const dl_p = kron(idim,0) * dx(i_p)
                                    + kron(idim,1) * dy(j_p)
                                    + kron(idim,2) * dz(k_p);

                    double const slope = slope_limiter(
                        (var(i_p, j_p, k_p) - var(i, j, k)) / ((dl + dl_p) / 2),
                        (var(i, j, k) - var(i_m, j_m, k_m)) / ((dl_m + dl) / 2));

                    var_rec(i, j, k, 0, idim) =  var(i, j, k) - (dl / 2) * slope;
                    var_rec(i, j, k, 1, idim) =  var(i, j, k) + (dl / 2) * slope;
                }
            });

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);

            std::string filename = "./exec_time_cudaEvent_face_reconstruction.dat";
            std::ofstream timing_file(filename, std::ios::app);
            if (timing_file) {
                timing_file << "cuda" << " " << ms << "\n";
            }

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

        } else {

            auto const [begin, end] = cell_range(range);
            int Nx = end[0] - begin[0];
            int Ny = end[1] - begin[1];
            int Nz = end[2] - begin[2];

            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < Nx; i += blockDim.x * gridDim.x) 
            {
                int ii = i + begin[0];
                for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < Ny; j += blockDim.y * gridDim.y) 
                {
                    int jj = j + begin[1];
                    for (int k = blockIdx.z * blockDim.z + threadIdx.z; k < Nz; k += blockDim.z * gridDim.z) 
                    {
                        int kk = k + begin[2];
                        for (int idim = 0; idim < ndim; ++idim)
                        {
                            auto const [i_m, j_m, k_m] = lindex(idim, ii, jj, kk); // i - 1
                            auto const [i_p, j_p, k_p] = rindex(idim, ii, jj, kk); // i + 1
                            double const dl   = kron(idim,0) * dx(ii)
                                            + kron(idim,1) * dy(jj)
                                            + kron(idim,2) * dz(kk);
                            double const dl_m = kron(idim,0) * dx(i_m)
                                            + kron(idim,1) * dy(j_m)
                                            + kron(idim,2) * dz(k_m);
                            double const dl_p = kron(idim,0) * dx(i_p)
                                            + kron(idim,1) * dy(j_p)
                                            + kron(idim,2) * dz(k_p);

                            double const slope = slope_limiter(
                                (var(i_p, j_p, k_p) - var(ii, jj, kk)) / ((dl + dl_p) / 2),
                                (var(ii, jj, kk) - var(i_m, j_m, k_m)) / ((dl_m + dl) / 2));

                            var_rec(ii, jj, kk, 0, idim) =  var(ii, jj, kk) - (dl / 2) * slope;
                            var_rec(ii, jj, kk, 1, idim) =  var(ii, jj, kk) + (dl / 2) * slope;
                        }
                    }
                }
            }
        }
    }
};

} // namespace novapp