#!/bin/bash

#SBATCH --job-name=sgpu_sizes
#SBATCH --output=%x.o%j
#SBATCH --time=02:00:00
#SBATCH -C h100

## GPU allocation
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

set -e

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

: > performance.dat

SIZES=("16 16 16" "16 16 32" "32 16 32"  "32 32 32" "32 32 64" "64 32 64" \
"64 64 64" "64 64 128" "128 64 128" "128 128 128" "128 128 256" \
"256 128 256" "256 256 256" "256 256 512")
# SIZES=("16 16 16" "16 16 32")

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

INI_FILE="./inputs/rayleigh_taylor3d.ini"

REPORT_DIR=./../reports_sizes
mkdir -p "$REPORT_DIR"

echo "Current directory: $(pwd)"

for size in "${SIZES[@]}"; do
    
    > performance.dat

    NX=$(echo $size | awk '{print $1}')
    NY=$(echo $size | awk '{print $2}')
    NZ=$(echo $size | awk '{print $3}')

    echo "Test with grid: Nx=$NX, Ny=$NY, Nz=$NZ"

    sed -i "s/^Nx_glob.*/Nx_glob = $NX/" $INI_FILE
    sed -i "s/^Ny_glob.*/Ny_glob = $NY/" $INI_FILE
    sed -i "s/^Nz_glob.*/Nz_glob = $NZ/" $INI_FILE

    REPORT_NAME="report_H100_${NX}_${NY}_${NZ}.txt"

    ./$BUILD_DIR/src/nova++ $INI_FILE --face-reconstruction="tiling__cudaEvent" > "$REPORT_DIR/$REPORT_NAME"

    echo "Summary of times for grid size $NX x $NY x $NZ:"
    awk '
    BEGIN { sum = 0 }
    NR <= 18 { sum += $4 }
    END {
        printf("   Sum of the first 18 lines: %f\n", sum)
    }' performance.dat

    echo ""
done

echo "All benchmarks have been executed."
echo "Reports are available in $REPORT_DIR/ from heraclespp_ers/"
