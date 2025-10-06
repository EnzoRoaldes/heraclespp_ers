// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file factory_face_reconstruction.hpp
//!

#pragma once

#include <memory>
#include <string>
#include <stdexcept>

#include <kokkos_shortcut.hpp> //


namespace novapp
{
std::unique_ptr<IFaceReconstruction> new_factory_face_reconstruction(std::string const& name, std::array<int, 3> tiling);
} // namespace novapp