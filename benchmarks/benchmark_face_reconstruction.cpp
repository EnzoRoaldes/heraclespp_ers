#include <benchmark/benchmark.h>

#include <PerfectGas.hpp>
#include <factory_face_reconstruction.hpp>
#include <memory>
#include <string>
#include <stdexcept>



namespace {

void set_constant_bytes_processed(benchmark::State& state, std::size_t const bytes)
{
    state.counters["bytes_per_second"] = benchmark::Counter(static_cast<double>(bytes), benchmark::Counter::kIsIterationInvariantRate);
}

void set_constant_cells_processed(benchmark::State& state, std::size_t const cells)
{
    state.counters["cells_per_second"] = benchmark::Counter(static_cast<double>(cells), benchmark::Counter::kIsIterationInvariantRate);
}






void FaceReconstruction(benchmark::State& state)
{   
    int const nx = state.range();
    int const ny = state.range();
    int const nz = state.range();
    novapp::Range const range({0, 0, 0}, {nx, ny, nz}, 0);
    novapp::Grid const grid({nx, ny, nz}, {1, 1, 1}); // à vérifier
    novapp::KV_cdouble_3d const rho("rho", nx, ny, nz);
    novapp::KV_cdouble_5d const rho_rec("rho_rec", nx, ny, nz); // passer en 5d

    Kokkos::deep_copy(rho, 1);
    Kokkos::deep_copy(rho_rec, 1);

    // Default implementation name
    std::string face_reconstruction_impl = "base";
    // Parse command line for --face-reconstruction=<name>
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--face-reconstruction=") == 0) {
            face_reconstruction_impl = arg.substr(strlen("--face-reconstruction="));
        }
    }

    std::unique_ptr<IFaceReconstruction> face_reconstruction = factory_face_reconstruction(face_reconstruction_impl, false);

    Kokkos::fence();
    for ([[maybe_unused]] auto _ : state) {
        face_reconstruction->execute(range, grid, rho, rho_rec);
        Kokkos::fence();
    }

    std::size_t const cells = (static_cast<std::size_t>(nx) * ny) * nz;

    set_constant_cells_processed(state, cells);
    set_constant_bytes_processed(state, ((2 + novapp::ndim) + (1 + novapp::ndim)) * cells);
}

} // namespace

BENCHMARK(FaceReconstruction)
