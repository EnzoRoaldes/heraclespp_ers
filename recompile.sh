#!/bin/bash

## PDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
##
## SPDX-License-Identifier: MIT

#SBATCH --job-name=build_heraclespp
#SBATCH --output=build_heraclespp_%j.out
#SBATCH --error=build_heraclespp_%j.err
#SBATCH --time=00:30:00
#SBATCH -C h100
## #SBATCH --qos=qos_gpu_h100-dev

## GPU allocation
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

# Effacer les dossiers
rm -rf build_H100/Hydro build_H100/nova++

set -e

## To clean and load modules defined at the compile and link phases
module purge
module load arch/h100
module load cmake/3.31.4
module load gcc/11.4.1
module load cuda/12.8.0
module load openmpi/4.1.5-cuda
module load hdf5/1.12.0-mpi-cuda


. vendor/install_pdi/share/pdi/env.sh

export KOKKOS_TOOLS_LIBS=/linkhome/rech/genmdl01/ult48qa/kokkos-tools/profiling/nvtx-connector/kp_nvtx_connector.so

# echo of commands
set -x

# Compile
cmake --build build_H100

