# set(SOURCES mpi_scope_guard.cpp ${CMAKE_SOURCE_DIR}/src/mpi_scope_guard.hpp)

# add_library(${CMAKE_SOURCE_DIR}/src/mpi_scope_guard.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(benchmarks)

target_sources(benchmarks
    PRIVATE
        benchmark_conv_prim_to_cons.cpp
        benchmark_conv_cons_to_prim.cpp
        benchmark_face_reconstruction.cpp
        benchmark_face_reconstruction.hpp
        main.cpp
)

target_link_libraries(benchmarks
    PRIVATE
        MPI::MPI_CXX
        benchmark::benchmark
        hydro
        utils
)