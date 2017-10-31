#!/bin/bash
# Sample Slurm Script for use with OpenMPI on Plato
# Begin Slurm directives with #SBATCH

EXECUTABLE=exe-name

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name=petsc_slurm_template

# ============================================================================

echo "Starting run at: `date`"
srun ./${EXECUTABLE}
echo "Program finished with exit code $? at: `date`"
exit 0
