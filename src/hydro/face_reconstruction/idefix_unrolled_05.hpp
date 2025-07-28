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
class FaceReconstructionIdefixUnrolled05 : public IFaceReconstruction
{

private:
    SlopeLimiter m_slope_limiter;

public:
    explicit FaceReconstructionIdefixUnrolled05(SlopeLimiter limiter) 
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

        KV_cdouble_1d const dx = grid.dx;
        KV_cdouble_1d const dy = grid.dy;
        KV_cdouble_1d const dz = grid.dz;
        
        auto const& slope_limiter = m_slope_limiter;
        
        auto const [begin, end] = cell_range(range);

        idefix_for(
            "face_reconstruction",
            begin[2],end[2],begin[1],end[1],begin[0],end[0], // idefix takes k then, j then i
            KOKKOS_LAMBDA(int i, int j, int k)
        {
            // IDIM=0
            {                
                double const slope = slope_limiter(      
                    (var(i+1, j, k) - var(i, j, k)) / ((dx(i) + dx(i+1)) * 0.5),
                    (var(i, j, k) - var(i-1, j, k)) / ((dx(i-1) + dx(i)) * 0.5));
            
                var_rec(i, j, k, 0, 0) =  var(i, j, k) - (dx(i) * 0.5) * slope;
                var_rec(i, j, k, 1, 0) =  var(i, j, k) + (dx(i) * 0.5) * slope;
            }


            // IDIM=1
            {
                double const slope = slope_limiter(      
                    (var(i, j+1, k) - var(i, j, k)) / ((dy(j) + dy(j+1)) * 0.5),
                    (var(i, j, k) - var(i, j-1, k)) / ((dy(j-1) + dy(j)) * 0.5));
            
                var_rec(i, j, k, 0, 1) =  var(i, j, k) - (dy(j) * 0.5) * slope;
                var_rec(i, j, k, 1, 1) =  var(i, j, k) + (dy(j) * 0.5) * slope;
            }


            // IDIM=2
            {
                double const slope = slope_limiter(      
                    (var(i, j, k+1) - var(i, j, k)) / ((dz(k) + dz(k+1)) * 0.5),
                    (var(i, j, k) - var(i, j, k-1)) / ((dz(k-1) + dz(k)) * 0.5));
            
                var_rec(i, j, k, 0, 2) =  var(i, j, k) - (dz(k) * 0.5) * slope;
                var_rec(i, j, k, 1, 2) =  var(i, j, k) + (dz(k) * 0.5) * slope;
            }

        });
    }
};

} // namespace novapp
