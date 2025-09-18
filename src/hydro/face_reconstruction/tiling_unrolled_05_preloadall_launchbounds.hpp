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
class FaceReconstructionTilingUnrolled05PreloadAllLaunchBounds : public IFaceReconstruction
{

private:
    SlopeLimiter m_slope_limiter;

public:
    explicit FaceReconstructionTilingUnrolled05PreloadAllLaunchBounds(SlopeLimiter limiter) 
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

        std::array<int, 2> m_launchbound = {32, 1}; // Default launchbound ?

        std::string filename = "./launchbound.dat";
        std::ifstream launchbound_file(filename);
        if (launchbound_file) {
            int lb_x, lb_y;
            launchbound_file >> lb_x >> lb_y;
            const_cast<std::array<int, 2>&>(m_launchbound) = {lb_x, lb_y};
            // printf("Using launchbound from %s: {%d, %d}\n", filename.c_str(), lb_x, lb_y);
        }
        else {
            // printf("%s not found, using default launchbound {%d, %d}\n", filename.c_str(), m_launchbound[0], m_launchbound[1]);
        }

        Kokkos::parallel_for(
            "face_reconstruction",
            cell_mdrange_tiling(range, m_tiling),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                const double var_ijk = var(i, j, k);

                double var_i1_jk = var(i+1, j, k);
                double var_1i_jk = var(i-1, j, k);
                double dx_i = dx(i);
                double dx_i1 = dx(i+1);
                double dx_1i = dx(i-1);

                double var_ij1_k = var(i, j+1, k);
                double var_i_1jk = var(i, j-1, k);
                double dy_j = dy(j);
                double dy_j1 = dy(j+1);
                double dy_1j = dy(j-1);

                double var_ij_k1 = var(i, j, k+1);
                double var_ij_1k = var(i, j, k-1);
                double dz_k = dz(k);
                double dz_k1 = dz(k+1);
                double dz_1k = dz(k-1);


                // IDIM=0
                {
                    double const slope = slope_limiter(
                        (var_i1_jk - var_ijk) / ((dx_i + dx_i1) * 0.5),
                        (var_ijk - var_1i_jk) / ((dx_1i + dx_i) * 0.5));

                    var_rec(i, j, k, 0, 0) = var_ijk - (dx_i * 0.5) * slope;
                    var_rec(i, j, k, 1, 0) = var_ijk + (dx_i * 0.5) * slope;
                }


                // IDIM=1
                {
                    double const slope = slope_limiter(
                        (var_ij1_k - var_ijk) / ((dy_j + dy_j1) * 0.5),
                        (var_ijk - var_i_1jk) / ((dy_1j + dy_j) * 0.5));

                    var_rec(i, j, k, 0, 1) =  var_ijk - (dy_j * 0.5) * slope;
                    var_rec(i, j, k, 1, 1) =  var_ijk + (dy_j * 0.5) * slope;
                }


                // IDIM=2
                {
                    double const slope = slope_limiter(
                        (var_ij_k1 - var_ijk) / ((dz_k + dz_k1) * 0.5),
                        (var_ijk - var_ij_1k) / ((dz_1k + dz_k) * 0.5));

                    var_rec(i, j, k, 0, 2) =  var_ijk - (dz_k * 0.5) * slope;
                    var_rec(i, j, k, 1, 2) =  var_ijk + (dz_k * 0.5) * slope;
                }
            }
        );
    }
};

} // namespace novapp