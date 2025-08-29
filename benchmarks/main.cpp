#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include <mpi.h>

int main(int argc, char** argv)
{
    ::Kokkos::ScopeGuard const scope(argc, argv);
    MPI_Init(&argc, &argv);

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    MPI_Finalize();
    ::benchmark::Shutdown();
    return 0;
}
