// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file extrapolation_reconstruction.hpp
//!

#pragma once

#include <memory>

#include <kokkos_shortcut.hpp>

namespace novapp
{

class Grid;
class Range;
class Gravity;

template <concepts::GravityField Gravity>
class IExtrapolationReconstruction
{
public:
    IExtrapolationReconstruction();

    IExtrapolationReconstruction(IExtrapolationReconstruction const& rhs);

    IExtrapolationReconstruction(IExtrapolationReconstruction&& rhs) noexcept;

    virtual ~IExtrapolationReconstruction() noexcept;

    IExtrapolationReconstruction& operator=(IExtrapolationReconstruction const& rhs);

    IExtrapolationReconstruction& operator=(IExtrapolationReconstruction&& rhs) noexcept;

    virtual void execute(
        Range const& range,
        Grid const& grid,
        Gravity const& gravity,
        double dt_reconstruction,
        KV_cdouble_6d const& u_rec,
        KV_cdouble_5d const& P_rec,
        KV_double_5d const& rho_rec,
        KV_double_6d const& rhou_rec,
        KV_double_5d const& E_rec,
        KV_double_6d const& fx_rec) const
        = 0;
};

} // namespace novapp
