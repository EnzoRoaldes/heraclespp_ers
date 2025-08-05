#!/bin/bash

## PDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
##
## SPDX-License-Identifier: MIT

#SBATCH --job-name=face_reconstruction
#SBATCH --output=%x.o%j
#SBATCH --time=02:00:00
#SBATCH -C h100

## GPU allocation
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

set -e

# Inform users about the --detailed flag
echo "Note: Use the --detailed flag if you need detailed reports."

# Parse arguments
DETAILED_REPORT=false
for arg in "$@"; do
    if [ "$arg" == "--detailed" ]; then
        DETAILED_REPORT=true
    fi
done

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

METHODS=("base" "idefix_05" "idefix" "idefix_unrolled_05_2" "idefix_unrolled_05_fma" "idefix_unrolled_05" "idefix_unrolled_05_varijk" "idefix_unrolled_dxyz_varijk" "idefix_unrolled_dxyz" "idefix_unrolled" "idefix_unrolled_preload_05" "idefix_unrolled_preloadall" "idefix_unrolled_preload" "tiling_05_varijk" "tiling_direct_mem" "tiling" "tiling_unrolled_05" "tiling_unrolled_05_varijk" "tiling_unrolled" "tiling_varijk")

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

REPORT_DIR=./../reports_versions
mkdir -p "$REPORT_DIR"

for method in "${METHODS[@]}"; do
    echo "### Testing face reconstruction: $method ###"

    OUTPUT_FILE="report_H100_${method}.ncu-rep"

    if [ "$DETAILED_REPORT" = true ]; then
        ncu --nvtx --import-source yes --target-processes all --set full \
            --kernel-name-base demangled -k regex:"FaceReconstruction*" \
            -f -o "$OUTPUT_FILE" ./$BUILD_DIR/src/nova++ \
            ./inputs/rayleigh_taylor3d.ini --face-reconstruction="$method"
    else
        ncu --nvtx --import-source yes --target-processes all --print-summary per-gpu \
            --kernel-name-base demangled -k regex:"FaceReconstruction*" \
            -f -o "$OUTPUT_FILE" ./$BUILD_DIR/src/nova++ \
            ./inputs/rayleigh_taylor3d.ini --face-reconstruction="$method"
    fi

    mv "$OUTPUT_FILE" "$REPORT_DIR/$OUTPUT_FILE"
done

if [ "$DETAILED_REPORT" = false ]; then
    echo "### Execution Times for All Kernels ###"
    for method in "${METHODS[@]}"; do
        OUTPUT_FILE="report_H100_${method}.ncu-rep"
        REPORT_PATH="$REPORT_DIR/$OUTPUT_FILE"

        if [[ -f "$REPORT_PATH" ]]; then
            average_time=$(ncu --import "$REPORT_PATH" --print-summary per-gpu | grep "Duration" | awk '{print $NF}')
            total_time=$(echo "$average_time * 18" | bc)
            echo "Method: $method | Average Kernel Time: $average_time | Invocations: 18 (fixed) | Total Time: $total_time"
        else
            echo "Report not found for method $method"
        fi
    done
fi