#pragma once

namespace benchmark { class State; }

// static void BM_FaceReconstruction_LB(benchmark::State& st);

void set_constant_bytes_processed(benchmark::State& state, std::size_t const bytes);

void set_constant_cells_processed(benchmark::State& state, std::size_t const cells);

void FaceReconstruction(benchmark::State& state, std::string const& method, int tx, int ty, int tz);

void RegisterVersionBenchmarks();

void RegisterTilingBenchmarks();

void RegisterIdefixTilingBenchmarks();

void RegisterDimensionBenchmarks();