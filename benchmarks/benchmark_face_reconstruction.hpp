#pragma once

namespace benchmark { class State; }

// static void BM_FaceReconstruction_LB(benchmark::State&);

void set_constant_bytes_processed(benchmark::State&, std::size_t const);

void set_constant_cells_processed(benchmark::State&, std::size_t const);

// void FaceReconstructionImpl(benchmark::State&, std::string const&, int, int, int);

void RegisterVersionBenchmarks();

void RegisterTilingBenchmarks();

void RegisterIdefixTilingBenchmarks();

void RegisterDimensionBenchmarks();