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
#SBATCH --job-name=sgpu_spe_tilingH100_face

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

# Initialize timing file
: > exec_time_cudaEvent_face_reconstruction.dat

# Parse flag for mode selection
MODE="cuda" # Default mode
if [[ "$1" == "--ncu" ]]; then
    MODE="ncu"
fi

# Compile only once
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


TILING=("16 2 2" "16 4 2" "32 1 2" "32 2 2") # Add more tiling configurations as needed

#  _----------_
# |  CASE NCU  |
#  '----------'

if [[ "$MODE" == "ncu" ]]; then
    echo -e "\nRunning in \"ncu\" mode"
    echo -e "\n  _----------_ \n |  CASE NCU  | \n  '----------'\n\n"

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


#  _-----------_
# |  CASE CUDA  |
#  '-----------'

elif [[ "$MODE" == "cuda" ]]; then
    echo -e "\nRunning in \"cuda\" mode"
    echo -e "\n  _-----------_ \n |  CASE CUDA  | \n  '-----------'\n\n"

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
        
        : > ./tiling.dat
        echo "$tx $ty $tz" > ./tiling.dat

        ./$BUILD_DIR/src/nova++ ./inputs/rayleigh_taylor3d.ini --face-reconstruction="tiling" --timer
    done

    # Search for the minimum and maximum total execution time (sum of last 17 runs for each tiling, skipping the first)
    FILENAME="./exec_time_cudaEvent_face_reconstruction.dat"
    # "./timing_face_reconstruction_tiling.dat" previous name
    if [ -f $FILENAME ]; then
        echo "Best and worst tiling (by sum of all runs) and total execution time:"
        awk '
        {
            key =  $3 " " $4 " " $5
            count[key] += 1
            vals[key, count[key]] = $2
            prod[key] = $3 * $4 * $5
        }
        END {
            minsum = -1
            maxsum = -1
            maxsum_1024 = -1
            default_sum = -1

            minkey = ""
            maxkey = ""
            maxkey_1024 = ""

            all_valid = 1

            for (k in count) {
                if (count[k] != 18) {
                    print "Error: Tiling", k, "does not have 18 occurrences. Found:", count[k]
                    all_valid = 0
                } else {
                    sum = 0
                    for (i = 1; i <= 18; ++i) {
                        sum += vals[k, i]
                    }
                    if (minsum < 0 || sum < minsum) {
                        minsum = sum
                        minkey = k
                    }
                    if (maxsum < 0 || sum > maxsum) {
                        maxsum = sum
                        maxkey = k
                    }
                    if (prod[k] == 1024) {
                        if (maxsum_1024 < 0 || sum > maxsum_1024) {
                            maxsum_1024 = sum
                            maxkey_1024 = k
                        }
                    }
                    if (k == "16 2 2") {
                        default_sum = sum
                    }
                }
            }
            if (all_valid == 0) {
                print "Error: Not all tilings have 18 occurrences. Exiting."
                exit 1
            }
            print "Best:", minkey, minsum
            print "Worst:", maxkey, maxsum
            print "Worst with product=1024:", maxkey_1024, maxsum_1024
            print "Default tiling: (16,2,2) ", default_sum
            if (default_sum > 0) {
                speedup = ((default_sum - minsum) / default_sum) * 100
                print "Speedup compared to default tiling (16,2,2):", speedup "%"
            } else {
                print "Error: Default tiling (16,2,2) not found."
            }
        }
        ' $FILENAME

    else
        echo "Error: No $FILENAME file found."
        exit 1
    fi

else 
    echo "Invalid mode specified. Use --ncu for NCU profiling or no flag for CUDA profiling."
    exit 1
fi