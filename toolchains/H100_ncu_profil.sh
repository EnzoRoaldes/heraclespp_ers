#!/bin/bash

## PDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
##
## SPDX-License-Identifier: MIT

#SBATCH --job-name=sgpu_ncuH100
#SBATCH --output=%x.o%j
#SBATCH --time=00:30:00
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

cd ..
. vendor/install_pdi/share/pdi/env.sh

export KOKKOS_TOOLS_LIBS=/linkhome/rech/genmdl01/ult48qa/kokkos-tools/profiling/nvtx-connector/kp_nvtx_connector.so

# echo of commands
set -x

## To compute in the submission directory
cd "${SLURM_SUBMIT_DIR}"

ncu --nvtx --import-source yes --target-processes all --launch-skip 6 --launch-count 6 --set full --kernel-name-base demangled -k regex:".*Linear.*" -f -o report_H100.ncu-rep ./../build_H100/src/nova++ ./../inputs/rayleigh_taylor3d.ini