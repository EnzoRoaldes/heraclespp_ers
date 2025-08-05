#!/bin/bash

## PDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
##
## SPDX-License-Identifier: MIT

#SBATCH --job-name=sgpu_ncuH100_specific
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

# Réinitialise le fichier de timing au début du benchmark
: > timing_face_reconstruction_specific.dat

TILING=("16 2 2" "16 4 2" "32 1 2" "32 2 2")

# Compile only once
BUILD_DIR=build_H100
cmake \
    -D CMAKE_BUILD_TYPE=Release \
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
    -B $BUILD_DIR
cmake --build $BUILD_DIR

REPORT_DIR=./../reports_specific_ncu
mkdir -p "$REPORT_DIR"

for tile in "${TILING[@]}"; do
    tx=$(echo "$tile" | cut -d' ' -f1)
    ty=$(echo "$tile" | cut -d' ' -f2)
    tz=$(echo "$tile" | cut -d' ' -f3)

    product=$((tx * ty * tz))
    if (( product > 512 )); then
        echo "Skipping tiling $tx $ty $tz (product > 512)"
        continue
    fi

    echo "### Profiling with tiling $tx $ty $tz ###"
    pwd
    : > ./tiling.dat
    echo "$tx $ty $tz" > ./tiling.dat

    REPORT_NAME="report_H100_specific_${tx}_${ty}_${tz}.ncu-rep"

    ncu --nvtx --import-source yes --target-processes all --print-summary per-gpu \
    --kernel-name-base demangled -k regex:"FaceReconstruction*" \
    -f -o "$REPORT_NAME" ./$BUILD_DIR/src/nova++ \
    ./inputs/rayleigh_taylor3d.ini --face-reconstruction="tiling"

    mv "$REPORT_NAME" "$REPORT_DIR/$REPORT_NAME"
done

# Extract and print execution times from reports
echo "### Execution Times for All Kernels ###"
for tile in "${TILING[@]}"; do
    tx=$(echo "$tile" | cut -d' ' -f1)
    ty=$(echo "$tile" | cut -d' ' -f2)
    tz=$(echo "$tile" | cut -d' ' -f3)

    REPORT_NAME="report_H100_specific_${tx}_${ty}_${tz}.ncu-rep"
    REPORT_PATH="$REPORT_DIR/$REPORT_NAME"
    
    if [[ -f "$REPORT_PATH" ]]; then
        average_time=$(ncu --import "$REPORT_PATH" --print-summary per-gpu | grep "Duration" | awk '{print $NF}')
        total_time=$(echo "$average_time * 18" | bc)
        echo "Tiling: $tx $ty $tz | Average Kernel Time: $average_time | Invocations: 18 (fixed) | Total Time: $total_time"
    else
        echo "Report not found for tiling $tx $ty $tz"
    fi
done