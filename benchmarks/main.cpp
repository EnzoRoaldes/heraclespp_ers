#include <mpi.h>

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include "face_reconstruction.hpp"
#include "benchmark_face_reconstruction.hpp"

int main(int argc, char** argv) 
{
    ::Kokkos::ScopeGuard const scope(argc, argv);
    MPI_Init(&argc, &argv);
    ::benchmark::Initialize(&argc, argv);

    RegisterVersionBenchmarks();
    RegisterTilingBenchmarks();
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