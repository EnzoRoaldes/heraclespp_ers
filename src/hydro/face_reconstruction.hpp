// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md fileAdd commentMore actions
//
// SPDX-License-Identifier: MIT

//!
//! @file face_reconstruction.hpp
//!

#pragma once

#include <cassert>
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

    virtual void execute(
        Range const& range,
        Grid const& grid,
        KV_cdouble_3d const& var,
        KV_double_5d const& var_rec) const = 0;
};

} // namespace novapp