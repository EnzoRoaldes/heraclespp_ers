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

#include "slope_limiters.hpp"

namespace novapp
{

class IFaceReconstruction
{
public:
    IFaceReconstruction() = default;

    IFaceReconstruction(IFaceReconstruction const& rhs) = default;

    IFaceReconstruction(IFaceReconstruction&& rhs) noexcept = default;

    virtual ~IFaceReconstruction() noexcept = default;

    IFaceReconstruction& operator=(IFaceReconstruction const& rhs) = default;

    IFaceReconstruction& operator=(IFaceReconstruction&& rhs) noexcept = default;

    //! @param[in] range output iteration range
    //! @param[in] grid provides grid information
    //! @param[in] var cell values
    //! @param[out] var_rec reconstructed values at interfaces
    virtual void execute(
        Range const& range,
        Grid const& grid,
        KV_cdouble_3d const& var,
        KV_double_5d const& var_rec) const
        = 0;
};

template <class SlopeLimiter>
class LimitedLinearReconstruction : public IFaceReconstruction
{
    static_assert(
            std::is_invocable_r_v<
            double,
            SlopeLimiter,
            double,
            double>,
            "Invalid slope limiter.");

private:
    SlopeLimiter m_slope_limiter;

public:
    explicit LimitedLinearReconstruction(SlopeLimiter const& slope_limiter)
        : m_slope_limiter(slope_limiter)
    {
    }

    void execute(
        Range const& range,
        Grid const& grid,
        KV_cdouble_3d const& var,
        KV_double_5d const& var_rec) const final
    {
        assert(equal_extents({0, 1, 2}, var, var_rec));
        assert(var_rec.extent(3) == 2);
        assert(var_rec.extent(4) == ndim);

        auto const& slope_limiter = m_slope_limiter;
        
        KV_cdouble_1d const dx = grid.dx;
        KV_cdouble_1d const dy = grid.dy;
        KV_cdouble_1d const dz = grid.dz;

        std::array<int, 3> m_tiling = {16, 2, 2}; // Default tiling
        


        
        
        
        
        std::ifstream tiling_file("tiling.dat");
        if (tiling_file) {
            int ti, tj, tk;
            tiling_file >> ti >> tj >> tk;
            const_cast<std::array<int, 3>&>(m_tiling) = {ti, tj, tk};
            printf("Using tiling from tiling.dat: {%d, %d, %d}\n", ti, tj, tk);
        }
        else {
            printf("tiling.dat not found, using default tiling\n");
        }

        Kokkos::Timer timer;

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

                double const slope = slope_limiter(
                    (var(i_p, j_p, k_p) - var(i, j, k)) / ((dl + dl_p) / 2),
                    (var(i, j, k) - var(i_m, j_m, k_m)) / ((dl_m + dl) / 2));

                var_rec(i, j, k, 0, idim) =  var(i, j, k) - (dl / 2) * slope;
                var_rec(i, j, k, 1, idim) =  var(i, j, k) + (dl / 2) * slope;
            }
        });

        Kokkos::fence("face_reconstruction");
        
        double time = timer.seconds()*1000000;
        timer.reset();

        // Append timing and tiling to a file (concurrent-safe)
        static std::mutex file_mutex;
        {
            std::lock_guard<std::mutex> lock(file_mutex);
            std::ofstream timing_file("timing_face_reconstruction.dat", std::ios::app | std::ios::out);
            timing_file << m_tiling[0] << " " << m_tiling[1] << " " << m_tiling[2] << " " << time << "\n";
        }
    }
};

inline std::unique_ptr<IFaceReconstruction> factory_face_reconstruction(
        std::string const& slope)
{
    if (slope == "Constant")
    {
        return std::make_unique<LimitedLinearReconstruction<Constant>>(Constant());
    }

    if (slope == "VanLeer")
    {
        return std::make_unique<LimitedLinearReconstruction<VanLeer>>(VanLeer());
    }

    if (slope == "Minmod")
    {
        return std::make_unique<LimitedLinearReconstruction<Minmod>>(Minmod());
    }

    if (slope == "VanAlbada")
    {
        return std::make_unique<LimitedLinearReconstruction<VanAlbada>>(VanAlbada());
    }

    throw std::runtime_error("Unknown face reconstruction algorithm: " + slope + ".");
}

} // namespace novapp