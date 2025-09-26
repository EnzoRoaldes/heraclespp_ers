// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file extrapolation_reconstruction.cpp
//!

#include <cassert>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <geom.hpp>
#include <grid.hpp>
#include <kokkos_shortcut.hpp>
#include <kronecker.hpp>
#include <ndim.hpp>
#include <range.hpp>

#include "euler_equations.hpp"
#include "source_terms.hpp"

namespace novapp
{

public:
    IExtrapolationReconstruction::IExtrapolationReconstruction() = default;

    IExtrapolationReconstruction::IExtrapolationReconstruction(IExtrapolationReconstruction const& rhs) = default;

    IExtrapolationReconstruction::IExtrapolationReconstruction(IExtrapolationReconstruction&& rhs) noexcept = default;

    IExtrapolationReconstruction::~IExtrapolationReconstruction() noexcept = default;

    IExtrapolationReconstruction& IExtrapolationReconstruction::operator=(IExtrapolationReconstruction const&) = default;

    IExtrapolationReconstruction& IExtrapolationReconstruction::operator=(IExtrapolationReconstruction&&) noexcept = default;
};