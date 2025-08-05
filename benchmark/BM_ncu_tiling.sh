#!/bin/bash

## PDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
##
## SPDX-License-Identifier: MIT

#SBATCH --job-name=sgpu_ncuH100
#SBATCH --output=%x.o%j
#SBATCH --time=10:00:00
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
: > timing_face_reconstruction_tiling.dat

I=("1" "2" "4" "8" "11" "16" "32" "64" "117" "128" "256" "512") # 1024 not supported by kokkos
J=("1" "2" "4" "8" "11" "16" "32" "64" "117" "128" "256" "512")
K=("1" "2" "4" "8" "11" "16" "32" "64")

# Compile only once with the first values
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

REPORT_DIR=./../reports_tiling_ncu
mkdir -p "$REPORT_DIR"

# Inform users about the --detailed flag
echo "Note: Use the --detailed flag if you need detailed reports."

# Parse arguments
DETAILED_REPORT=false
for arg in "$@"; do
    if [ "$arg" == "--detailed" ]; then
        DETAILED_REPORT=true
    fi
done

for i in "${I[@]}"; do
    for j in "${J[@]}"; do
        for k in "${K[@]}"; do
            
            if (( i * j * k > 512 )); then # cuda can go up to 1024, but kokkos does not support it
                echo "Skipping tiling $i $j $k (i*j*k >= 512)"
                continue
            fi

            echo "### Profiling with tiling $i $j $k ###"

            echo "$(pwd)"

            : > tiling.dat
            echo "$i $j $k" > tiling.dat

            REPORT_NAME="report_H100_tiling_${i}_${j}_${k}.ncu-rep"

            if [ "$DETAILED_REPORT" = true ]; then
                ncu --nvtx --import-source yes --target-processes all --set full \
                    --kernel-name-base demangled -k regex:"FaceReconstruction*" \
                    -f -o "$REPORT_NAME" ./$BUILD_DIR/src/nova++ \
                    ./inputs/rayleigh_taylor3d.ini --face-reconstruction="tiling"
            else
                ncu --nvtx --import-source yes --target-processes all --print-summary per-gpu \
                    --kernel-name-base demangled -k regex:"FaceReconstruction*" \
                    -f -o "$REPORT_NAME" ./$BUILD_DIR/src/nova++ \
                    ./inputs/rayleigh_taylor3d.ini --face-reconstruction="tiling"
            fi

            mv "$REPORT_NAME" "$REPORT_DIR/$REPORT_NAME"
        done
    done
done

if [ "$DETAILED_REPORT" = false ]; then
    echo "### Execution Times for All Kernels ###"
    min_time=""
    max_time=""
    total_time_16_2_2=""
    total_time_1_1_1=""
    for i in "${I[@]}"; do
        for j in "${J[@]}"; do
            for k in "${K[@]}"; do
                if (( i * j * k > 512 )); then
                    continue
                fi

                REPORT_NAME="report_H100_tiling_${i}_${j}_${k}.ncu-rep"
                REPORT_PATH="$REPORT_DIR/$REPORT_NAME"

                if [[ -f "$REPORT_PATH" ]]; then
                    average_time=$(ncu --import "$REPORT_PATH" --print-summary per-gpu | grep "Duration" | awk '{print $NF}')
                    total_time=$(echo "$average_time * 18" | bc)

                    # Update min and max times
                    if [ -z "$min_time" ] || (( $(echo "$total_time < $min_time" | bc -l) )); then
                        min_time=$total_time
                    fi

                    if [ -z "$max_time" ] || (( $(echo "$total_time > $max_time" | bc -l) )); then
                        max_time=$total_time
                    fi

                    # Store total times for specific tilings
                    if [ "$i" -eq 16 ] && [ "$j" -eq 2 ] && [ "$k" -eq 2 ]; then
                        total_time_16_2_2=$total_time
                    fi

                    if [ "$i" -eq 32 ] && [ "$j" -eq 2 ] && [ "$k" -eq 2 ]; then
                        total_time_32_2_2=$total_time
                    fi

                    echo "Tiling: $i $j $k | Average Kernel Time: $average_time | Invocations: 18 (fixed) | Total Time: $total_time"
                else
                    echo "Report not found for tiling $i $j $k"
                fi
            done
        done
    done

    echo "Minimum Total Time: $min_time"
    echo "Maximum Total Time: $max_time"
    echo "Total Time for Tiling 16_2_2: $total_time_16_2_2"
    echo "Total Time for Tiling 32_2_2: $total_time_32_2_2"
fi