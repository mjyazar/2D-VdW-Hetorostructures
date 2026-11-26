#!/bin/bash
#PBS -P hh39
#PBS -N GrapheneHeterostructure
#PBS -q normal
#PBS -l walltime=24:00:00
#PBS -l mem=576GB
#PBS -l jobfs=200GB
#PBS -l ncpus=144
#PBS -l software=espresso
#PBS -l storage=scratch/hh39
#PBS -l wd

set -euo pipefail

module load python3/3.12.13  
module load qe/7.3

cd /scratch/hh39/mj0450/2D-VdW-Hetorostructures

# cctivate Python env
source /scratch/hh39/mj0450/venv/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export QE_NPROC="$PBS_NCPUS"
export QE_NK=6
export QE_PP_NK=48
export MPLBACKEND=Agg

mkdir -p log

echo "Job: $PBS_JOBID"
echo "CPUs: $PBS_NCPUS"
echo "Commit: $(git rev-parse HEAD)"

echo -e "\nRUNNING PROGRAM"
python3 -u main.py 2>&1 | tee "log/run_${PBS_JOBID}.log"
