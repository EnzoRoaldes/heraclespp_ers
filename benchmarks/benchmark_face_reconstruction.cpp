#include <benchmark/benchmark.h>

#include <mpi.h>
#include <PerfectGas.hpp>
#include <face_reconstruction.hpp>
#include <face_reconstruction_factory.hpp>
#include <memory>
#include <string>
#include <stdexcept>

// a vérifier
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <range.hpp>
#include <grid.hpp>
#include <grid_type.hpp> // sûr
#include <nova_params.hpp>



namespace {

void set_constant_bytes_processed(benchmark::State& state, std::size_t const bytes)
{
    state.counters["bytes_per_second"] = benchmark::Counter(static_cast<double>(bytes)
        , benchmark::Counter::kIsIterationInvariantRate);
}

void set_constant_cells_processed(benchmark::State& state, std::size_t const cells)
{
    state.counters["cells_per_second"] = benchmark::Counter(static_cast<double>(cells)
        , benchmark::Counter::kIsIterationInvariantRate);
}








void FaceReconstruction(benchmark::State& state)
{   
    int const nx = state.range();
    int const ny = state.range();
    int const nz = state.range();
    novapp::Range const range({0, 0, 0}, {nx, ny, nz}, 0);

    // construction de la grid
    double const xmin = 0;
    double const xmax = 1;
    double const ymin = 0;
    double const ymax = 1;
    double const zmin = 0;
    double const zmax = 1;

    INIReader const reader;
    novapp::Param param(reader);
    param.xmin = xmin;
    param.xmax = xmax;
    param.ymin = ymin;
    param.ymax = ymax;
    param.zmin = zmin;
    param.zmax = zmax;
    for(int idim = 0; idim < novapp::ndim; ++idim)
    {
        param.Nx_glob_ng[idim] = 15; // a parametrer
    }

    novapp::Grid grid(param);
    std::unique_ptr const grid_type = std::make_unique<novapp::Regular>(std::array {xmin, ymin, zmin}
        , std::array {xmax, ymax, zmax});

    novapp::KDV_double_1d x_glob("x_glob", grid.Nx_glob_ng[0]+2*grid.Nghost[0]+1);
    novapp::KDV_double_1d y_glob("y_glob", grid.Nx_glob_ng[1]+2*grid.Nghost[1]+1);
    novapp::KDV_double_1d z_glob("z_glob", grid.Nx_glob_ng[2]+2*grid.Nghost[2]+1);
    grid_type->execute(grid.Nghost, grid.Nx_glob_ng, x_glob.view_host()
        , y_glob.view_host(), z_glob.view_host());
    novapp::modify_host(x_glob, y_glob, z_glob);
    novapp::sync_device(x_glob, y_glob, z_glob);
    grid.set_grid(x_glob.view_device(), y_glob.view_device(), z_glob.view_device());
    //////

    // THOMAS VOULAIT INITIALISER VAR_IJK À X + Y + Z (x(i), y(j), z(k) ?)

    novapp::KV_double_3d rho("rho", nx, ny, nz);
    novapp::KV_double_5d rho_rec("rho_rec", nx, ny, nz, 2, novapp::ndim);
    printf("Views created\n");

    Kokkos::deep_copy(rho, 1);
    Kokkos::deep_copy(rho_rec, -1);
    printf("Data initialized\n");

    std::unique_ptr<novapp::IFaceReconstruction> face_reconstruction 
        = novapp::factory_face_reconstruction("base", false);
    printf("Face reconstruction created\n");
    Kokkos::fence();
    for ([[maybe_unused]] auto _ : state) {
        printf("Face reconstruction executing\n");
        face_reconstruction->execute(range, grid, rho, rho_rec);
        printf("Face reconstruction executed\n");
        Kokkos::fence();
    }

    std::size_t const cells = (static_cast<std::size_t>(nx) * ny) * nz;

    set_constant_cells_processed(state, cells);

    // ici fonctionne pour "base"
    set_constant_bytes_processed(state, sizeof(double) * (novapp::ndim*(3*3 + 6)
        + novapp::ndim*(2)) * cells); // adapter
}

} // namespace

BENCHMARK(FaceReconstruction)->DenseRange(8, 63, 8)->DenseRange(64, 320, 32);
