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

: > exec_time_cudaEvent_face_reconstruction.dat

I=("1" "2" "4" "8" "16" "32" "64" "128" "256" "512") # 1024 not supported by kokkos
J=("1" "2" "4" "8" "16" "32" "64" "128" "256" "512")
K=("1" "2" "4" "8" "16" "32" "64")

# I=("4" "8" "16" "32") # 1024 not supported by kokkos
# J=("1" "2" "4")
# K=("1" "2" "4")

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

for i in "${I[@]}"; do
    for j in "${J[@]}"; do
        for k in "${K[@]}"; do
            
            if (( i * j * k > 512 )); then # cuda can go up to 1024, but kokkos does not support it
                echo "Skipping tiling $i $j $k (i*j*k >= 512)"
                continue
            fi

            echo "### Profiling with tiling $i $j $k ###"

            : > ./tiling.dat
            echo "$i $j $k" > ./tiling.dat

            ./$BUILD_DIR/src/nova++ ./inputs/rayleigh_taylor3d.ini --face-reconstruction="tiling" --timer
        done
    done
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