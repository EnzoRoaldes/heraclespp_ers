// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file factory_extrapolation_reconstruction.hpp
//!

#pragma once

#include <memory>
#include <string>
#include <stdexcept>

#include <kokkos_shortcut.hpp>
#include <gravity.hpp>
#include <PerfectGas.hpp>

namespace novapp
{
std::unique_ptr<IExtrapolationReconstruction<UniformGravity>> new_factory_extrapolation_reconstruction(std::string const& name, thermodynamics::PerfectGas const& eos, std::array<int, 3> tiling);
} // namespace novapp