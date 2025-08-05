#pragma once
#include "face_reconstruction.hpp"
#include "face_reconstruction/base.hpp"
#include "face_reconstruction/tiling.hpp"
#include "face_reconstruction/tiling__std_chrono.hpp"
#include "face_reconstruction/tiling__cudaEvent.hpp"
#include "face_reconstruction/tiling_varijk.hpp"
#include "face_reconstruction/tiling_unrolled.hpp"
#include "face_reconstruction/tiling_unrolled_05.hpp"
#include "face_reconstruction/tiling_unrolled_05_varijk.hpp"
#include "face_reconstruction/tiling_direct_mem.hpp"
#include "face_reconstruction/tiling_05_varijk.hpp"

#include "face_reconstruction/idefix.hpp"
#include "face_reconstruction/idefix_unrolled.hpp"
#include "face_reconstruction/idefix_unrolled_preloadall.hpp"
#include "face_reconstruction/idefix_unrolled_preload.hpp"
#include "face_reconstruction/idefix_unrolled_preload_05.hpp"
#include "face_reconstruction/idefix_unrolled_dxyz.hpp"
#include "face_reconstruction/idefix_unrolled_dxyz_varijk.hpp"
#include "face_reconstruction/idefix_unrolled_05.hpp"
#include "face_reconstruction/idefix_unrolled_05_varijk.hpp"
#include "face_reconstruction/idefix_unrolled_05_fma.hpp"
#include "face_reconstruction/idefix_unrolled_05_2.hpp"
#include "face_reconstruction/idefix_05.hpp"

#include <memory>
#include <string>
#include <stdexcept>

namespace novapp {

inline std::unique_ptr<IFaceReconstruction> factory_face_reconstruction(std::string const& name, bool enable_timer)
{
    if (name == "base") return std::make_unique<FaceReconstructionBase<Minmod>>(Minmod(), enable_timer);
    if (name == "tiling") return std::make_unique<FaceReconstructionTiling<Minmod>>(Minmod(), enable_timer);
    if (name == "tiling__std_chrono") return std::make_unique<FaceReconstructionTilingStdChrono<Minmod>>(Minmod());
    if (name == "tiling__cudaEvent") return std::make_unique<FaceReconstructionTilingCudaEvent<Minmod>>(Minmod());
    if (name == "tiling_varijk") return std::make_unique<FaceReconstructionTilingVarijk<Minmod>>(Minmod(), enable_timer);
    if (name == "tiling_unrolled") return std::make_unique<FaceReconstructionTilingUnrolled<Minmod>>(Minmod(), enable_timer);
    if (name == "tiling_unrolled_05") return std::make_unique<FaceReconstructionTilingUnrolled05<Minmod>>(Minmod(), enable_timer);
    if (name == "tiling_unrolled_05_varijk") return std::make_unique<FaceReconstructionTilingUnrolled05Varijk<Minmod>>(Minmod(), enable_timer);
    if (name == "tiling_direct_mem") return std::make_unique<FaceReconstructionTilingDirectMem<Minmod>>(Minmod(), enable_timer);
    if (name == "tiling_05_varijk") return std::make_unique<FaceReconstructionTiling05Varijk<Minmod>>(Minmod(), enable_timer);
    if (name == "idefix") return std::make_unique<FaceReconstructionIdefix<Minmod>>(Minmod(), enable_timer);
    if (name == "idefix_unrolled") return std::make_unique<FaceReconstructionIdefixUnrolled<Minmod>>(Minmod(), enable_timer);
    if (name == "idefix_unrolled_preloadall") return std::make_unique<FaceReconstructionIdefixUnrolledPreloadAll<Minmod>>(Minmod(), enable_timer);
    if (name == "idefix_unrolled_preload") return std::make_unique<FaceReconstructionIdefixUnrolledPreload<Minmod>>(Minmod(), enable_timer);
    if (name == "idefix_unrolled_preload_05") return std::make_unique<FaceReconstructionIdefixUnrolledPreload05<Minmod>>(Minmod(), enable_timer);
    if (name == "idefix_unrolled_dxyz") return std::make_unique<FaceReconstructionIdefixUnrolledDxyz<Minmod>>(Minmod(), enable_timer);
    if (name == "idefix_unrolled_dxyz_varijk") return std::make_unique<FaceReconstructionIdefixUnrolledDxyzVarijk<Minmod>>(Minmod(), enable_timer);
    if (name == "idefix_unrolled_05") return std::make_unique<FaceReconstructionIdefixUnrolled05<Minmod>>(Minmod(), enable_timer);
    if (name == "idefix_unrolled_05_varijk") return std::make_unique<FaceReconstructionIdefixUnrolled05Varijk<Minmod>>(Minmod(), enable_timer);
    if (name == "idefix_unrolled_05_fma") return std::make_unique<FaceReconstructionIdefixUnrolled05Fma<Minmod>>(Minmod(), enable_timer);
    if (name == "idefix_unrolled_05_2") return std::make_unique<FaceReconstructionIdefixUnrolled052<Minmod>>(Minmod(), enable_timer);
    if (name == "idefix_05") return std::make_unique<FaceReconstructionIdefix05<Minmod>>(Minmod(), enable_timer);
    throw std::runtime_error("Unknown face reconstruction implementation: " + name);
}

} // namespace novapp
