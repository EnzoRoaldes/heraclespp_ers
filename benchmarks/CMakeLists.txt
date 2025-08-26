find_package(benchmark REQUIRED)

add_executable(benchmarks)

target_sources(benchmarks 
               PRIVATE  benchmark_conv_cons_to_prim.cpp 
                        benchmark_face_reconstruction.cpp 
                        main.cpp)

target_link_libraries(benchmarks 
                      PRIVATE   benchmark::benchmark 
                                hydro)
