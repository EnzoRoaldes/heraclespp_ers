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
#include <Kokkos_Profiling_ScopedRegion.hpp>
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
class FaceReconstructionIdefixUnrolledDxyz : public IFaceReconstruction
{

private:
    SlopeLimiter m_slope_limiter;

public:
    explicit FaceReconstructionIdefixUnrolledDxyz(SlopeLimiter limiter) 
        : m_slope_limiter(limiter) {}

    void execute(
        Range const& range,
        Grid const& grid,
        KV_cdouble_3d const& var,
        KV_double_5d const& var_rec) const override
    {
        assert(equal_extents({0, 1, 2}, var, var_rec));
        assert(var_rec.extent(3) == 2);
        assert(var_rec.extent(4) == ndim);
        
        auto const& slope_limiter = m_slope_limiter;
        
        Kokkos::Profiling::pushRegion("face_reconstruction");

        auto grid_dx = grid.dx;
        auto grid_dy = grid.dy;
        auto grid_dz = grid.dz;
        
        auto const [begin, end] = cell_range(range);
        
        
        KV_double_1d dx("dx_scaled", end[0] - begin[0]+1);
        Kokkos::parallel_for("scale_dx", end[0] - begin[0]+1, KOKKOS_LAMBDA(const int i) 
        {
            dx(i) = 0.5 * grid_dx(i);
        });

        KV_double_1d dy("dy_scaled", end[1] - begin[1]+1);
        Kokkos::parallel_for("scale_dy", end[1] - begin[1]+1, KOKKOS_LAMBDA(const int i) 
        {
            dy(i) = 0.5 * grid_dy(i);
        });

        KV_double_1d dz("dz_scaled", end[2] - begin[2]+1);
        Kokkos::parallel_for("scale_dz", end[2] - begin[2]+1, KOKKOS_LAMBDA(const int i) 
        {
            dz(i) = 0.5 * grid_dz(i);
        });

        idefix_for(
            "idefix_for",
            begin[0],end[0],begin[1],end[1],begin[2],end[2],
            KOKKOS_LAMBDA(int i, int j, int k)
        {

            // IDIM=0
            {                
                double const slope = slope_limiter(      
                    (var(i+1, j, k) - var(i, j, k)) / (dx(i) + dx(i+1)),
                    (var(i, j, k) - var(i-1, j, k)) / (dx(i-1) + dx(i)));
            
                var_rec(i, j, k, 0, 0) =  var(i, j, k) - dx(i) * slope;
                var_rec(i, j, k, 1, 0) =  var(i, j, k) + dx(i) * slope;
            }


            // IDIM=1
            {
                double const slope = slope_limiter(      
                    (var(i, j+1, k) - var(i, j, k)) / (dy(j) + dy(j+1)),
                    (var(i, j, k) - var(i, j-1, k)) / (dy(j-1) + dy(j)));
            
                var_rec(i, j, k, 0, 1) =  var(i, j, k) - dy(j) * slope;
                var_rec(i, j, k, 1, 1) =  var(i, j, k) + dy(j) * slope;
            }


            // IDIM=2
            {
                double const slope = slope_limiter(      
                    (var(i, j, k+1) - var(i, j, k)) / (dz(k) + dz(k+1)),
                    (var(i, j, k) - var(i, j, k-1)) / (dz(k-1) + dz(k)));
            
                var_rec(i, j, k, 0, 2) =  var(i, j, k) - dz(k) * slope;
                var_rec(i, j, k, 1, 2) =  var(i, j, k) + dz(k) * slope;
            }
        });

        Kokkos::Profiling::popRegion();
    }
};

} // namespace novapp
