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
class FaceReconstructionTilingDirectMem : public IFaceReconstruction
{

private:
    SlopeLimiter m_slope_limiter;

public:
    explicit FaceReconstructionTilingDirectMem(SlopeLimiter limiter) 
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

        auto *ptr_var = var.data();
        auto *ptr_var_rec = var_rec.data();

        auto s1 = var_rec.stride(1);
        auto s2 = var_rec.stride(2);
        auto s3 = var_rec.stride(3);
        auto s4 = var_rec.stride(4);    

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
                for (int idim = 0; idim < ndim; ++idim)
                {
                    auto const [i_m, j_m, k_m] = lindex(idim, i, j, k); // i - 1
                    auto const [i_p, j_p, k_p] = rindex(idim, i, j, k); // i + 1
                    double const dl = kron(idim,0) * dx(i)
                                    + kron(idim,1) * dy(j)
                                    + kron(idim,2) * dz(k);
                    double const dl_m = kron(idim,0) * dx(i_m)
                                        + kron(idim,1) * dy(j_m)
                                        + kron(idim,2) * dz(k_m);
                    double const dl_p = kron(idim,0) * dx(i_p)
                                        + kron(idim,1) * dy(j_p)
                                        + kron(idim,2) * dz(k_p);

                    auto const offset = i + j * s1 + k * s2;

                    double const slope = slope_limiter(
                    ( *(ptr_var + i_p + j_p*s1 + k_p*s2) - *(ptr_var + offset) ) / ((dl + dl_p) * 0.5),
                    ( *(ptr_var + offset) - *(ptr_var + i_m + j_m*s1 + k_m*s2) ) / ((dl_m + dl) * 0.5));    
                    
                    *(ptr_var_rec + offset + 0*s3 + idim*s4) = *(ptr_var + offset) - (dl * 0.5) * slope;
                    *(ptr_var_rec + offset + 1*s3 + idim*s4) = *(ptr_var + offset) + (dl * 0.5) * slope;
                }
            }
        );
    }
};

} // namespace novapp