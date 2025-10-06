// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <string>
#include <fstream>
#include <extrapolation_reconstruction.hpp>
#include <factory_extrapolation_reconstruction.hpp>
#include <benchmark/benchmark.h>
#include "benchmark_extrapolation_reconstruction.hpp"
#include "parallel_for.hpp"
#include <grid.hpp>
#include <grid_type.hpp>
#include <gravity.hpp>
#include <PerfectGas.hpp>
#include <int_cast.hpp>
#include <kokkos_shortcut.hpp>

#include <ndim.hpp>
#include <range.hpp>

namespace benchmark_extrapolation_reconstruction {

void set_constant_bytes_processed(benchmark::State& state, std::size_t const bytes)
{
    state.counters["bytes_per_second"] = benchmark::Counter(static_cast<double>(bytes), benchmark::Counter::kIsIterationInvariantRate);
}

void set_constant_cells_processed(benchmark::State& state, std::size_t const cells)
{
    state.counters["cells_per_second"] = benchmark::Counter(static_cast<double>(cells), benchmark::Counter::kIsIterationInvariantRate);
}

void ExtrapolationReconstructionImpl(benchmark::State& state, std::string const& method, int tx, int ty, int tz)
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

    novapp::thermodynamics::PerfectGas const eos(1.4, 1.0); // gamma, mu ?

    novapp::Grid grid(Nx_glob_ng, mpi_dims_cart, Ng);
    novapp::Regular const grid_type(std::array {xmin, ymin, zmin}, std::array {xmax, ymax, zmax});

    novapp::KDV_double_1d x_glob("x_glob", grid.Nx_glob_ng[0] + 2 * grid.Nghost[0] + 1);
    novapp::KDV_double_1d y_glob("y_glob", grid.Nx_glob_ng[1] + 2 * grid.Nghost[1] + 1);
    novapp::KDV_double_1d z_glob("z_glob", grid.Nx_glob_ng[2] + 2 * grid.Nghost[2] + 1);
    grid_type.execute(grid.Nghost, grid.Nx_glob_ng, x_glob.view_host(), y_glob.view_host(), z_glob.view_host());
    novapp::modify_host(x_glob, y_glob, z_glob);
    novapp::sync_device(x_glob, y_glob, z_glob);
    grid.set_grid(x_glob.view_device(), y_glob.view_device(), z_glob.view_device());

    novapp::KV_cdouble_6d const u_rec("u_rec", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, novapp::ndim, novapp::ndim);
    novapp::KV_double_5d const P_rec("P_rec", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, novapp::ndim);
    novapp::KV_double_5d const rho_rec("rho_rec", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, novapp::ndim);
    novapp::KV_double_6d const rhou_rec("rhou_rec", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, novapp::ndim, novapp::ndim);
    novapp::KV_double_5d const E_rec("E_rec", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, novapp::ndim);
    novapp::KV_double_6d const fx_rec("fx_rec", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, novapp::ndim, param.nfx=1);

    Kokkos::deep_copy(u_rec, -1);
    Kokkos::deep_copy(P_rec, -1);
    Kokkos::deep_copy(rho_rec, -1);
    Kokkos::deep_copy(rhou_rec, -1);
    Kokkos::deep_copy(E_rec, -1);
    Kokkos::deep_copy(fx_rec, -1);


    //
    std::unique_ptr<UniformGravity> g;
    g = std::make_unique<UniformGravity>(make_gravity(param, grid, rho.view_device()));
    dt_reconstruction = dt/2;
    //


    std::array<int, 3> tiling = {tx, ty, tz};

    std::unique_ptr<novapp::IExtrapolationReconstruction<UniformGravity>> 
    const extrapolation_reconstruction = novapp::new_factory_extrapolation_reconstruction(method, eos, tiling);

     // void execute(
    //     Range const& range, OK
    //     Grid const& grid, OK
    //     Gravity const& gravity, A VOIR
    //     double const dt_reconstruction, A VOIR
    //     KV_cdouble_6d const& u_rec, OK
    //     KV_cdouble_5d const& P_rec, OK
    //     KV_double_5d const& rho_rec, OK
    //     KV_double_6d const& rhou_rec, OK
    //     KV_double_5d const& E_rec, OK
    //     KV_double_6d const& fx_rec, OK

    novapp::Range const range = novapp::grid.range.no_ghosts();
    Kokkos::fence();
    for ([[maybe_unused]] auto _ : state) {
        extrapolation_reconstruction->execute(range, grid, gravity, dt_reconstruction,
            u_rec, P_rec, rho_rec, rhou_rec, E_rec, fx_rec);
        Kokkos::fence();
    }


    std::size_t const cells = (static_cast<std::size_t>(nx) * ny) * nz;
    
    set_constant_cells_processed(state, cells);
    
    // set_constant_bytes_processed(state, sizeof(double) * (1 + novapp::ndim * 2) * cells); A MODIFIER
}



// ---------------- Tests ----------------

// 1) Test "version" : comparaison des différentes versions
void RegisterVersionBenchmarks() {
    std::vector<std::string> const methods = {
        "base",
    };

    for (auto const& method : methods) {
        std::string name = "version_extra/" + method;
        ::benchmark::RegisterBenchmark(
            name.c_str(),
            [method](benchmark::State& st) {
                // tx,ty,tz à 0 par défaut ; si method=="tiling", le fichier tiling.dat sera écrit.
                benchmark_extrapolation_reconstruction::ExtrapolationReconstructionImpl(st, method, 0, 0, 0);
            }
        )->Arg(320);
    }
}


} // namespace benchmark_extrapolation_reconstruction