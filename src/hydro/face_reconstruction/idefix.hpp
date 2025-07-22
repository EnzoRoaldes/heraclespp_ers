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
	    
        
        Kokkos::Timer timer;

        
        auto const [begin, end] = cell_range(range);

        idefix_for(
            "face_reconstruction",
            begin[2],end[2],begin[1],end[1],begin[0],end[0], // idefix takes k then, j then i
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

        double time = timer.seconds()*1000000; // Convert to microseconds
        timer.reset();

        static std::mutex file_mutex;
        {
            std::lock_guard<std::mutex> lock(file_mutex);
            std::ofstream timing_file("timing_face_reconstruction_idefix.dat", std::ios::app | std::ios::out);
            timing_file << time << "\n";
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
