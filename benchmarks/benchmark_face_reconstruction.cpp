#include <mpi.h>

#include <fstream>
#include <string>

#include <benchmark/benchmark.h>

#include <face_reconstruction.hpp>
#include <factory_face_reconstruction.hpp>
#include <grid.hpp>
#include <grid_type.hpp>
#include <int_cast.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <range.hpp>


namespace {

std::vector<std::string> const methods = {
    "base",
    "idefix", "idefix_05", "idefix_unrolled_05_2", "idefix_unrolled_05_fma",
    "idefix_unrolled_05", "idefix_unrolled_05_varijk", "idefix_unrolled",
    "idefix_unrolled_preload_05", "idefix_unrolled_preloadall", "idefix_unrolled_preload",
    "tiling_05_varijk", "tiling_direct_mem", "tiling", "tiling_unrolled_05",
    "tiling_unrolled_05_varijk", "tiling_unrolled", "tiling_varijk",
    "tp_TeamThread", "tp_TeamThreadMDR"
};

void set_constant_bytes_processed(benchmark::State& state, std::size_t const bytes)
{
    state.counters["bytes_per_second"] = benchmark::Counter(static_cast<double>(bytes), benchmark::Counter::kIsIterationInvariantRate);
}

void set_constant_cells_processed(benchmark::State& state, std::size_t const cells)
{
    state.counters["cells_per_second"] = benchmark::Counter(static_cast<double>(cells), benchmark::Counter::kIsIterationInvariantRate);
}

void FaceReconstruction(benchmark::State& state, std::string const& method, int tx, int ty, int tz)
{
    int const nx = novapp::int_cast<int>(state.range());
    int const ny = nx;
    int const nz = nx;

    double const xmin = 0;
    double const xmax = 1;
    double const ymin = 0;
    double const ymax = 1;
    double const zmin = 0;
    double const zmax = 1;

    std::array<int, 3> const Nx_glob_ng {nx, ny, nz};
    std::array<int, 3> const mpi_dims_cart {0, 0, 0};
    int const Ng = 1;

    novapp::Grid grid(Nx_glob_ng, mpi_dims_cart, Ng);
    std::unique_ptr const grid_type = std::make_unique<novapp::Regular>(std::array {xmin, ymin, zmin}, std::array {xmax, ymax, zmax});

    novapp::KDV_double_1d x_glob("x_glob", grid.Nx_glob_ng[0] + 2 * grid.Nghost[0] + 1);
    novapp::KDV_double_1d y_glob("y_glob", grid.Nx_glob_ng[1] + 2 * grid.Nghost[1] + 1);
    novapp::KDV_double_1d z_glob("z_glob", grid.Nx_glob_ng[2] + 2 * grid.Nghost[2] + 1);
    grid_type->execute(grid.Nghost, grid.Nx_glob_ng, x_glob.view_host(), y_glob.view_host(), z_glob.view_host());
    novapp::modify_host(x_glob, y_glob, z_glob);
    novapp::sync_device(x_glob, y_glob, z_glob);
    grid.set_grid(x_glob.view_device(), y_glob.view_device(), z_glob.view_device());

    novapp::KV_double_3d const rho("rho", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]);
    novapp::KV_double_5d const rho_rec("rho_rec", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, novapp::ndim);

    Kokkos::deep_copy(rho, 1);
    Kokkos::deep_copy(rho_rec, -1);

    if (method == "tiling") {
        std::ofstream file("tiling.dat");
        file << tx << " " << ty << " " << tz << "\n";
    }

    std::unique_ptr<novapp::IFaceReconstruction> const face_reconstruction = novapp::factory_face_reconstruction(method, false);
    novapp::Range const range = grid.range.no_ghosts();
    Kokkos::fence();
    for ([[maybe_unused]] auto _ : state) {
        face_reconstruction->execute(range, grid, rho, rho_rec);
        Kokkos::fence();
    }

    std::size_t const cells = (static_cast<std::size_t>(nx) * ny) * nz;

    set_constant_cells_processed(state, cells);

    set_constant_bytes_processed(state, sizeof(double) * (1 + novapp::ndim * 2) * cells);
}

} // namespace



// ---------------- Tests ----------------
// 1) Test "version" : enregistre toutes les méthodes pour les temps d'exécution
void RegisterVersionBenchmarks() {
    for (auto const& method : methods) {
        std::string name = "version/" + method;
        ::benchmark::RegisterBenchmark(
            name.c_str(),
            [method](benchmark::State& st) {
                // tx,ty,tz à 0 par défaut ; si method=="tiling", le fichier tiling.dat sera écrit.
                FaceReconstruction(st, method, 0, 0, 0);
            }
        )->Arg(320);
    }
}

// 2) Test "tiling" : balayage (tx,ty,tz) pour la méthode "tiling"
void RegisterTilingBenchmarks() {
    std::vector<int> I = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    std::vector<int> J = I;
    std::vector<int> K = {1, 2, 4, 8, 16, 32, 64};
    for (int tx : I) {
        for (int ty : J) {
            for (int tz : K) {
                if (1LL * tx * ty * tz > 512) continue;
                std::string name = "tiling/Tx" + std::to_string(tx)
                                  + "_Ty" + std::to_string(ty)
                                  + "_Tz" + std::to_string(tz);
                ::benchmark::RegisterBenchmark(
                    name.c_str(),
                    [tx, ty, tz](benchmark::State& st) { FaceReconstruction(st, "tiling", tx, ty, tz); }
                )->Arg(320);
            }
        }
    }
}

// 3) Test "dimension" : balayage de la taille de grille pour la méthode "base"
void RegisterDimensionBenchmarks() {
    for (int n = 64; n <= 352; n += 32) {
        std::string name = "dimension/base/N" + std::to_string(n);
        ::benchmark::RegisterBenchmark(
            name.c_str(),
            [n](benchmark::State& st) {
                    FaceReconstruction(st, "base", 0, 0, 0);
                }
        )->Arg(n);
    }
}