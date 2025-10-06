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

# P6: 4 GPU H100 + RAM 80 GB, 96 CPU + RAM 468 GB
#SBATCH --constraint=h100

# Name of the job
#SBATCH --job-name=sgpu_dimensionH100_face

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

: > ./exec_time_cudaEvent_face_reconstruction.dat

# DIMENSIONS=("258 130 258" "258 516 65" "215 516 78") # à modifier

# Read dimensions from rayleigh_taylor3d.ini
INI_FILE="./inputs/rayleigh_taylor3d.ini"
NX=$(awk -F'=' '/^Nx_glob/ {gsub(/ /, "", $2); print $2}' $INI_FILE)
NY=$(awk -F'=' '/^Ny_glob/ {gsub(/ /, "", $2); print $2}' $INI_FILE)
NZ=$(awk -F'=' '/^Nz_glob/ {gsub(/ /, "", $2); print $2}' $INI_FILE)

# Add 2 to each dimension
NX=$((NX + 2))
NY=$((NY + 2))
NZ=$((NZ + 2))

PRODUCT=$((NX * NY * NZ))

# Compute all combinations of X, Y, Z such that X * Y * Z = PRODUCT and X, Y, Z >= 32
DIMENSIONS=()
for X in $(seq 32 $NX); do
    if ((PRODUCT % X == 0)); then
        for Y in $(seq 32 $NY); do
            if (((PRODUCT / X) % Y == 0)); then
                Z=$((PRODUCT / (X * Y)))
                if ((Z >= 32)); then
                    DIMENSIONS+=("$X $Y $Z")
                fi
            fi
        done
    fi
done

# echo "Computed dimensions: ${DIMENSIONS[@]}"

# Allow BUILD_DIR to be set from the command line, default to build_H100
BUILD_DIR=${BUILD_DIR:-build_H100}
if [ "$BUILD_DIR" = "build_H100" ]; then
    echo -e "Using default BUILD_DIR: $BUILD_DIR\n   use --export=BUILD_DIR=your_build_dir to set a different one."
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

for dim in "${DIMENSIONS[@]}"; do
    echo "### Testing dimension: $dim ###"

    : > ./dimension.dat
    echo "$dim" > ./dimension.dat

    ./$BUILD_DIR/src/nova++ ./inputs/rayleigh_taylor3d.ini --face-reconstruction="tp_TeamVectorMDR" --timer # tp_TeamVector
done

FILENAME="./exec_time_cudaEvent_face_reconstruction.dat"
if [ -f $FILENAME ]; then
    echo "Summary of execution times for different dimensions:"
    awk '
    {
        key = $3 " " $4 " " $5
        time = $2
        count[key] += 1
        vals[key, count[key]] = time
    }
    END {
        all_valid = 1
        min_time = -1
        min_key = ""

        for (k in count) {
            if (count[k] != 18) {
                print "Error: Dimension", k, "does not have 18 occurrences. Found:", count[k]
                all_valid = 0
            } else {
                sum = 0
                for (i = 1; i <= count[k]; ++i) {
                    sum += vals[k, i]
                }
                print "Dimension:", k, "Total execution time:", sum, "ms"

                if (min_time < 0 || sum < min_time) {
                    min_time = sum
                    min_key = k
                }
            }
        }

        if (all_valid == 0) {
            print "Error: Not all dimensions have 18 occurrences. Exiting."
            exit 1
        }

        print "Best dimension:", min_key, "with total execution time:", min_time, "ms"
    }
    ' $FILENAME
else
    echo "Error: No $FILENAME file found."
    exit 1
fi
