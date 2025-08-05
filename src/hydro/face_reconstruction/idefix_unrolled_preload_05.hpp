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

#include "../idefix_utils.hpp"
#include "../face_reconstruction.hpp"
#include "slope_limiters.hpp"




namespace novapp
{

template <typename SlopeLimiter>
class FaceReconstructionIdefixUnrolledPreload05 : public IFaceReconstruction
{

private:
    SlopeLimiter m_slope_limiter;
    bool m_enable_timer;

public:
    explicit FaceReconstructionIdefixUnrolledPreload05(SlopeLimiter limiter, bool enable_timer = false) 
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
        
        auto const [begin, end] = cell_range(range);

        if (m_enable_timer) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            
            idefix_for(
                "face_reconstruction",
                begin[2],end[2],begin[1],end[1],begin[0],end[0], // idefix takes k then, j then i
                KOKKOS_LAMBDA(int i, int j, int k)
                {
                    
                    double var_ijk = var(i,j,k);

                    // IDIM=0
                    {   
                        double var_i1_jk = var(i+1, j, k);
                        double var_1i_jk = var(i-1, j, k);
                        double dx_i = dx(i);
                        double dx_i1 = dx(i+1);
                        double dx_1i = dx(i-1);

                        double const slope = slope_limiter(      
                            (var_i1_jk - var_ijk) / ((dx_i + dx_i1) * 0.5),
                            (var_ijk - var_1i_jk) / ((dx_1i + dx_i) * 0.5));
                                
                        var_rec(i, j, k, 0, 0) =  var_ijk - (dx_i * 0.5) * slope;
                        var_rec(i, j, k, 1, 0) =  var_ijk + (dx_i * 0.5) * slope;
                    }

                    // IDIM=1
                    {
                        double var_ij1_k = var(i, j+1, k);
                        double var_i_1jk = var(i, j-1, k);
                        double dy_j = dy(j);
                        double dy_j1 = dy(j+1);
                        double dy_1j = dy(j-1);

                        double const slope = slope_limiter(      
                            (var_ij1_k - var_ijk) / ((dy_j + dy_j1) * 0.5),
                            (var_ijk - var_i_1jk) / ((dy_1j + dy_j) * 0.5));
                    
                        var_rec(i, j, k, 0, 1) =  var_ijk - (dy_j * 0.5) * slope;
                        var_rec(i, j, k, 1, 1) =  var_ijk + (dy_j * 0.5) * slope;
                    }

                    // IDIM=2
                    {
                        double var_ij_k1 = var(i, j, k+1);
                        double var_ij_1k = var(i, j, k-1);
                        double dz_k = dz(k);
                        double dz_k1 = dz(k+1);
                        double dz_1k = dz(k-1);

                        double const slope = slope_limiter(      
                            (var_ij_k1 - var_ijk) / ((dz_k + dz_k1) * 0.5),
                            (var_ijk - var_ij_1k) / ((dz_1k + dz_k) * 0.5));
                    
                        var_rec(i, j, k, 0, 2) =  var_ijk - (dz_k * 0.5) * slope;
                        var_rec(i, j, k, 1, 2) =  var_ijk + (dz_k * 0.5) * slope;
                    }
                }
            );

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);

            std::string filename = "./exec_time_cudaEvent_face_reconstruction.dat";
            std::ofstream timing_file(filename, std::ios::app);
            if (timing_file) {
                timing_file << "idefix_unrolled_preload_05" << " " << ms << "\n";
            }
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);

        } else {

            idefix_for(
                "face_reconstruction",
                begin[2],end[2],begin[1],end[1],begin[0],end[0], // idefix takes k then, j then i
                KOKKOS_LAMBDA(int i, int j, int k)
                {
                    
                    double var_ijk = var(i,j,k);

                    // IDIM=0
                    {   
                        double var_i1_jk = var(i+1, j, k);
                        double var_1i_jk = var(i-1, j, k);
                        double dx_i = dx(i);
                        double dx_i1 = dx(i+1);
                        double dx_1i = dx(i-1);

                        double const slope = slope_limiter(      
                            (var_i1_jk - var_ijk) / ((dx_i + dx_i1) * 0.5),
                            (var_ijk - var_1i_jk) / ((dx_1i + dx_i) * 0.5));
                                
                        var_rec(i, j, k, 0, 0) =  var_ijk - (dx_i * 0.5) * slope;
                        var_rec(i, j, k, 1, 0) =  var_ijk + (dx_i * 0.5) * slope;
                    }

                    // IDIM=1
                    {
                        double var_ij1_k = var(i, j+1, k);
                        double var_i_1jk = var(i, j-1, k);
                        double dy_j = dy(j);
                        double dy_j1 = dy(j+1);
                        double dy_1j = dy(j-1);

                        double const slope = slope_limiter(      
                            (var_ij1_k - var_ijk) / ((dy_j + dy_j1) * 0.5),
                            (var_ijk - var_i_1jk) / ((dy_1j + dy_j) * 0.5));
                    
                        var_rec(i, j, k, 0, 1) =  var_ijk - (dy_j * 0.5) * slope;
                        var_rec(i, j, k, 1, 1) =  var_ijk + (dy_j * 0.5) * slope;
                    }

                    // IDIM=2
                    {
                        double var_ij_k1 = var(i, j, k+1);
                        double var_ij_1k = var(i, j, k-1);
                        double dz_k = dz(k);
                        double dz_k1 = dz(k+1);
                        double dz_1k = dz(k-1);

                        double const slope = slope_limiter(      
                            (var_ij_k1 - var_ijk) / ((dz_k + dz_k1) * 0.5),
                            (var_ijk - var_ij_1k) / ((dz_1k + dz_k) * 0.5));
                    
                        var_rec(i, j, k, 0, 2) =  var_ijk - (dz_k * 0.5) * slope;
                        var_rec(i, j, k, 1, 2) =  var_ijk + (dz_k * 0.5) * slope;
                    }
                }
            );
        }
    }
};

} // namespace novapp
