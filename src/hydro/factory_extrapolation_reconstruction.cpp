// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file factory_extrapolation_reconstruction.cpp
//!

#include <memory>
#include <string>
#include <stdexcept>

#include "extrapolation_reconstruction.hpp"
#include "factory_extrapolation_reconstruction.hpp"

#include "extrapolation_reconstruction/base.hpp"


namespace novapp {
std::unique_ptr<IExtrapolationReconstruction<Gravity>> new_factory_extrapolation_reconstruction(std::string const& name, EOS const& eos)
{
    if (name == "base") return std::make_unique<ExtrapolationReconstructionBase<EOS, UniformGravity>>(eos);

    throw std::runtime_error("Unknown extrapolation reconstruction implementation: " + name);
}

} // namespace novapp