#pragma once

namespace benchmark { class State; }

namespace benchmark_extrapolation_reconstruction {

void set_constant_bytes_processed(benchmark::State& state, std::size_t const bytes);

void set_constant_cells_processed(benchmark::State& state, std::size_t const cells);

void ExtrapolationReconstructionImpl(benchmark::State& state, std::string const& method, int tx, int ty, int tz);

void RegisterVersionBenchmarks();

// void RegisterTilingBenchmarks();

// void RegisterDimensionBenchmarks();

} // namespace benchmark_extrapolation_reconstruction