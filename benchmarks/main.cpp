// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <mpi.h>

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

#include "benchmark_face_reconstruction.hpp"

int main(int argc, char** argv)
{
    ::Kokkos::ScopeGuard const scope(argc, argv);
    MPI_Init(&argc, &argv);
    ::benchmark::Initialize(&argc, argv);

    RegisterVersionBenchmarks();
    RegisterTilingBenchmarks();
    RegisterIdefixTilingBenchmarks();
    RegisterDimensionBenchmarks();
    // RegisterLaunchBoundsBenchmarks();

    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        MPI_Finalize();
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    MPI_Finalize();
    ::benchmark::Shutdown();
    return 0;
}
