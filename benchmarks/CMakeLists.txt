# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(benchmarks)
target_sources(
    benchmarks
    PRIVATE
        benchmark_conv_prim_to_cons.cpp
        benchmark_conv_cons_to_prim.cpp
        benchmark_face_reconstruction.cpp
        benchmark_face_reconstruction.hpp
        benchmark_extrapolation_reconstruction.cpp
        benchmark_extrapolation_reconstruction.hpp
        main.cpp
        ../src/hydro/factory_face_reconstruction.cpp
        ../src/hydro/factory_extrapolation_reconstruction.cpp
)

target_link_libraries(
    benchmarks
    PRIVATE MPI::MPI_CXX benchmark::benchmark hydro utils thermodynamics gravity
)
