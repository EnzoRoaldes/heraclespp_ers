#!/bin/bash

## PDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
##
## SPDX-License-Identifier: MIT

#SBATCH --job-name=sgpu_ncuH100
#SBATCH --output=%x.o%j
#SBATCH --time=02:00:00
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

: > ./exec_time_cudaEvent_face_reconstruction.dat

METHODS=("base" \
         "idefix" "idefix_05" "idefix_unrolled_05_2" "idefix_unrolled_05_fma" \
         "idefix_unrolled_05" "idefix_unrolled_05_varijk" "idefix_unrolled" \
         "idefix_unrolled_preload_05" "idefix_unrolled_preloadall" "idefix_unrolled_preload" \
         "tiling_05_varijk" "tiling_direct_mem" "tiling" "tiling_unrolled_05" \
         "tiling_unrolled_05_varijk" "tiling_unrolled" "tiling_varijk")
# METHODS=("base")

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

for method in "${METHODS[@]}"; do
    echo "### Profiling method: $method ###"

    REPORT_NAME="report_H100_${method}.ncu-rep"

    ./$BUILD_DIR/src/nova++ ./inputs/rayleigh_taylor3d.ini --face-reconstruction="$method" --timer

done

FILENAME="./exec_time_cudaEvent_face_reconstruction.dat"
if [ -f $FILENAME ]; then
    echo "Summary of execution times for different methods:"
    awk '
    {
        key = $1
        time = $2
        count[key] += 1
        vals[key, count[key]] = time
    }
    END {
        all_valid = 1
        min_time = -1
        min_key = ""
        base_time = -1

        for (k in count) {
            if (count[k] != 18) {
                print "Error: Method", k, "does not have 18 occurrences. Found:", count[k]
                all_valid = 0
            } else {
                sum = 0
                for (i = 1; i <= count[k]; ++i) {
                    sum += vals[k, i]
                }
                print "Method:", k, "Total execution time:", sum, "ms"

                if (k == "base") {
                    base_time = sum
                }

                if (min_time < 0 || sum < min_time) {
                    min_time = sum
                    min_key = k
                }
            }
        }

        if (all_valid == 0) {
            print "Error: Not all methods have 18 occurrences. Exiting."
            exit 1
        }

        if (base_time > 0) {
            speedup = (1 - min_time / base_time)*100
            print "Best method:", min_key, "with total execution time:", min_time, "ms"
            print "Speedup compared to base:", speedup "%"
        } else {
            print "Error: Base method not found. Cannot calculate speedup."
        }
    }
    ' $FILENAME
else
    echo "Error: No $FILENAME file found."
    exit 1
fi

