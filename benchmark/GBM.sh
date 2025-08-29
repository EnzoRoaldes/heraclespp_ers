#!/bin/bash

## PDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
##
## SPDX-License-Identifier: MIT

#SBATCH --job-name=google_BM_H100
#SBATCH --output=%x.o%j
#SBATCH --time=01:00:00
#SBATCH -C h100

## GPU allocation
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

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

BUILD_DIR=build_H100

# Remove the benchmarks executable if it exists
if [ -f "./$BUILD_DIR/benchmarks/benchmarks" ]; then
    echo "Removing existing $BUILD_DIR/benchmarks/benchmarks executable..."
    rm -f "./$BUILD_DIR/benchmarks/benchmarks"
fi

cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=20 \
    -D CMAKE_CXX_COMPILER=$PWD/vendor/kokkos/bin/nvcc_wrapper \
    -D Kokkos_ARCH_ICX=ON \
    -D Kokkos_ENABLE_DEPRECATED_CODE_4=OFF \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Novapp_SETUP=rayleigh_taylor3d  \
    -D Novapp_NDIM=3 \
    -D Novapp_EOS=PerfectGas \
    -D Novapp_GRAVITY=Uniform \
    -D Novapp_GEOM=Cartesian \
    -D Kokkos_ENABLE_CUDA=ON \
    -D Kokkos_ARCH_HOPPER90=ON \
    -D Kokkos_ENABLE_DEBUG=ON \
    -D BENCHMARK_ENABLE_TESTING=OFF \
    -B $BUILD_DIR
cmake --build $BUILD_DIR

./$BUILD_DIR/benchmarks/benchmarks