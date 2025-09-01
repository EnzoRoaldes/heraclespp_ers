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
std::unique_ptr<IFaceReconstruction> factory_face_reconstruction(std::string const& name, bool enable_timer);
} // namespace novapp