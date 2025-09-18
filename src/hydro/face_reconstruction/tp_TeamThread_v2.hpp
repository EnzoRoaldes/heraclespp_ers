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

#include "../face_reconstruction.hpp"
#include "slope_limiters.hpp"

namespace novapp
{

template <typename SlopeLimiter>
class FaceReconstructionTPTT2 : public IFaceReconstruction
{

private:
    SlopeLimiter m_slope_limiter;

public:
    explicit FaceReconstructionTPTT2(SlopeLimiter limiter)
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

        // Define the grid dimensions
        const auto [begin, end] = cell_range(range);

        const int bi = begin[0];
        const int bj = begin[1];
        const int bk = begin[2];

        int Ni = end[0] - bi;
        int Nj = end[1] - bj;
        int Nk = end[2] - bk;

        std::array<int, 3> dimension = {Ni, Nj, Nk};
        
        std::string filename = "./dimension.dat";
        std::ifstream dimension_file(filename);
        if (dimension_file) {
            int di, dj, dk;
            dimension_file >> di >> dj >> dk;
            dimension = {di, dj, dk};
            // printf("Using dimension from %s: {%d, %d, %d}\n", filename.c_str(), di, dj, dk);
        } else {
            // printf("%s not found, using default dimension {%d, %d, %d}\n", filename.c_str(), dimension[0], dimension[1], dimension[2]);
        }

        using team_policy = Kokkos::TeamPolicy<>;
        using member_type = team_policy::member_type;

        team_policy policy(Ni * Nj, Kokkos::AUTO);

        printf("Team size: %d\n", policy.team_size());

        Kokkos::parallel_for(
            "face_reconstruction",
            policy,
            KOKKOS_LAMBDA(const member_type& teamMember) {
                const int k = teamMember.league_rank(); // une équipe = une valeur de k
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(teamMember, Ni * Nj),
                    [=] (const int t) {

                        int j = t / Ni;
                        int i = t % Ni;

                        int ii = bi + i;
                        int jj = bj + j;
                        int kk = bk + k;

                        for (int idim = 0; idim < ndim; ++idim) {
                            auto const [i_m, j_m, k_m] = lindex(idim, ii, jj, kk); // ii - 1
                            auto const [i_p, j_p, k_p] = rindex(idim, ii, jj, kk); // ii + 1

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
                );
            }
        );
    }
};

} // namespace novapp