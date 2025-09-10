#!/bin/bash

## PDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
##
## SPDX-License-Identifier: MIT

# Number of nodes
#SBATCH --nodes=1

# Number of tasks (MPI processes)
#SBATCH --ntasks=1

# Number of cpu per task (Ex. OpenMP threads per MPI process)
#SBATCH --cpus-per-task=24

# Number of gpus
#SBATCH --gres=gpu:1

# Walltime for the job
#SBATCH --time=00:30:00

## Slurm partitions (for specific jobs)
##SBATCH --partition=compil

# P6: 4 GPU H100 + RAM 80 GB, 96 CPU + RAM 468 GB
#SBATCH --constraint=h100

# Name of the job
#SBATCH --job-name=sgpu_ncuH100

# Standard output (stdout)
#SBATCH --output=%x.%J.out

set -e

## To clean and load modules defined at the compile and link phases
module purge
module load arch/h100
module load cmake/3.31.4
module load gcc/11.4.1
module load cuda/12.8.0
module load openmpi/4.1.5-cuda
module load hdf5/1.12.0-mpi-cuda

cd /lustre/fswork/projects/rech/nnp/ult48qa/heraclespp_ers/

. vendor/install_pdi/share/pdi/env.sh

export KOKKOS_TOOLS_LIBS=/linkhome/rech/genmdl01/ult48qa/kokkos-tools/profiling/nvtx-connector/kp_nvtx_connector.so

# Allow BUILD_DIR to be set from the command line, default to build_H100
BUILD_DIR=${BUILD_DIR:-build_H100}
if [ "$BUILD_DIR" = "build_H100" ]; then
    echo -e "Using default BUILD_DIR: $BUILD_DIR\n   use --export=BUILD_DIR=your_build_dir to set a different one."
fi

# Remove the benchmarks directory if it exists
if [ -d "./$BUILD_DIR/benchmarks" ]; then
    echo "Removing existing ./$BUILD_DIR/benchmarks directory..."
    rm -rf "./$BUILD_DIR/benchmarks"
fi

cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=20 \
    -D CMAKE_CXX_COMPILER=$PWD/vendor/kokkos/bin/nvcc_wrapper \
    -D Kokkos_ARCH_ICX=ON \
    -D Kokkos_ENABLE_DEPRECATED_CODE_4=OFF \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Novapp_SETUP=rayleigh_taylor3d \
    -D Novapp_NDIM=3 \
    -D Novapp_EOS=PerfectGas \
    -D Novapp_GRAVITY=Uniform \
    -D Novapp_GEOM=Cartesian \
    -D Kokkos_ENABLE_CUDA=ON \
    -D Kokkos_ARCH_HOPPER90=ON \
    -D Kokkos_ENABLE_DEBUG=ON \
    -D BENCHMARK_ENABLE_TESTING=OFF \
    -D BENCHMARK_FORMAT=CSV \
    -D Novapp_BUILD_BENCHMARKING=ON \
    -B $BUILD_DIR

cmake --build $BUILD_DIR -j 24

./$BUILD_DIR/benchmarks/benchmarks --benchmark_filter="$1"