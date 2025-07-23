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

#include "slope_limiters.hpp"

// 3D loop
template <typename Function>
inline void idefix_for(const std::string & NAME,
                       const int & KB, const int & KE,
                       const int & JB, const int & JE,
                       const int & IB, const int & IE,
                       Function function) {
  // Kokkos 1D Range
    const int NK = KE - KB;
    const int NJ = JE - JB;
    const int NI = IE - IB;
    const int NKNJNI = NK*NJ*NI;
    const int NJNI = NJ * NI;
    Kokkos::parallel_for(NAME,NKNJNI,
      KOKKOS_LAMBDA (const int& IDX) {
        int k = IDX / NJNI;
        int j = (IDX - k*NJNI) / NI;
        int i = IDX - k*NJNI - j*NI;
        k += KB;
        j += JB;
        i += IB;
        function(i,j,k);
});}


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
	     		
        auto const [begin, end] = cell_range(range);

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
                    (var_i1_jk - var_ijk) / ((dx_i + dx_i1) / 2),
                    (var_ijk - var_1i_jk) / ((dx_1i + dx_i) / 2));
            
                var_rec(i, j, k, 0, 0) =  var_ijk - (dx_i / 2) * slope;
                var_rec(i, j, k, 1, 0) =  var_ijk + (dx_i / 2) * slope;
            }


            // IDIM=1
            {
                double var_ij1_k = var(i, j+1, k);
                double var_i_1jk = var(i, j-1, k);
                double dy_j = dy(j);
                double dy_j1 = dy(j+1);
                double dy_1j = dy(j-1);

                double const slope = slope_limiter(      
                    (var_ij1_k - var_ijk) / ((dy_j + dy_j1) / 2),
                    (var_ijk - var_i_1jk) / ((dy_1j + dy_j) / 2));
            
                var_rec(i, j, k, 0, 1) =  var_ijk - (dy_j / 2) * slope;
                var_rec(i, j, k, 1, 1) =  var_ijk + (dy_j / 2) * slope;
            }


            // IDIM=2
            {
                double var_ij_k1 = var(i, j, k+1);
                double var_ij_1k = var(i, j, k-1);
                double dz_k = dz(k);
                double dz_k1 = dz(k+1);
                double dz_1k = dz(k-1);

                double const slope = slope_limiter(      
                    (var_ij_k1 - var_ijk) / ((dz_k + dz_k1) / 2),
                    (var_ijk - var_ij_1k) / ((dz_1k + dz_k) / 2));
            
                var_rec(i, j, k, 0, 2) =  var_ijk - (dz_k / 2) * slope;
                var_rec(i, j, k, 1, 2) =  var_ijk + (dz_k / 2) * slope;
            }

        });
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
