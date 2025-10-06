// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file factory_face_reconstruction.cpp
//!

#include <memory>
#include <string>
#include <stdexcept>

#include <eos.hpp> // 
#include "concepts.hpp"
#include "extrapolation_reconstruction.hpp"
#include "extrapolation_reconstruction/base.hpp"
#include "factory_extrapolation_reconstruction.hpp"



namespace novapp {

// <EOS, Gravity> = <PerfectGas, Uniform>
std::unique_ptr<IExtrapolationReconstruction<UniformGravity>> new_factory_extrapolation_reconstruction(std::string const& name, thermodynamics::PerfectGas const& eos, std::array<int, 3> tiling= {16, 2, 2})
{
    if (name == "base") return std::make_unique<HancockExtrapolationReconstructionBase>(eos); // ou utliser : EOS const& eos(param.gamma, param.mu) ?
    throw std::runtime_error("Unknown extrapolation_reconstruction implementation: " + name);
}



} // namespace novapp