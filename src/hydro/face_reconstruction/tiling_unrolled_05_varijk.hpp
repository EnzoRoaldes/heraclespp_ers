// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md fileAdd commentMore actions
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

#include "../face_reconstruction.hpp"
#include "slope_limiters.hpp"

namespace novapp
{

template <typename SlopeLimiter>
class FaceReconstructionTilingUnrolled05Varijk : public IFaceReconstruction
{

private:
    SlopeLimiter m_slope_limiter;

public:
    explicit FaceReconstructionTilingUnrolled05Varijk(SlopeLimiter limiter) 
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

        std::array<int, 3> m_tiling = {16, 2, 2}; // Default tiling

        std::ifstream tiling_file("../tiling.dat");
        if (tiling_file) {
            int ti, tj, tk;
            tiling_file >> ti >> tj >> tk;
            const_cast<std::array<int, 3>&>(m_tiling) = {ti, tj, tk};
            printf("Using tiling from ../tiling.dat: {%d, %d, %d}\n", ti, tj, tk);
        }
        else {
            printf("../tiling.dat not found, using default tiling {%d, %d, %d}\n", m_tiling[0], m_tiling[1], m_tiling[2]);
        }

        Kokkos::parallel_for(
            "face_reconstruction",
            cell_mdrange(range, m_tiling),
            KOKKOS_LAMBDA(int i, int j, int k)
        {
            const double var_ijk = var(i, j, k);

            // IDIM=0
            {                
                double const slope = slope_limiter(      
                    (var(i+1, j, k) - var_ijk) / ((dx(i) + dx(i+1)) * 0.5),
                    (var_ijk - var(i-1, j, k)) / ((dx(i-1) + dx(i)) * 0.5));
            
                var_rec(i, j, k, 0, 0) =  var_ijk - (dx(i) * 0.5) * slope;
                var_rec(i, j, k, 1, 0) =  var_ijk + (dx(i) * 0.5) * slope;
            }


            // IDIM=1
            {
                double const slope = slope_limiter(      
                    (var(i, j+1, k) - var_ijk) / ((dy(j) + dy(j+1)) * 0.5),
                    (var_ijk - var(i, j-1, k)) / ((dy(j-1) + dy(j)) * 0.5));
            
                var_rec(i, j, k, 0, 1) =  var_ijk - (dy(j) * 0.5) * slope;
                var_rec(i, j, k, 1, 1) =  var_ijk + (dy(j) * 0.5) * slope;
            }


            // IDIM=2
            {
                double const slope = slope_limiter(      
                    (var(i, j, k+1) - var_ijk) / ((dz(k) + dz(k+1)) * 0.5),
                    (var_ijk - var(i, j, k-1)) / ((dz(k-1) + dz(k)) * 0.5));
            
                var_rec(i, j, k, 0, 2) =  var_ijk - (dz(k) * 0.5) * slope;
                var_rec(i, j, k, 1, 2) =  var_ijk + (dz(k) * 0.5) * slope;
            }
        });
    }
};

} // namespace novapp