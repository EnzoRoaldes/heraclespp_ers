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

#include <kokkos_shortcut.hpp> //


namespace novapp 
{
std::unique_ptr<IExtrapolationReconstruction<Gravity>> new_factory_extrapolation_reconstruction(std::string const& name, EOS const& eos);
} // namespace novapp