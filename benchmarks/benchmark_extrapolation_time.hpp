#pragma once

namespace benchmark { class State; }

void set_constant_bytes_processed(benchmark::State&, std::size_t const);

void set_constant_cells_processed(benchmark::State&, std::size_t const);
