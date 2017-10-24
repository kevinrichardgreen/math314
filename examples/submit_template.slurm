#!/bin/bash
# Sample Slurm Script for use with OpenMPI on Plato
# Begin Slurm directives with #SBATCH

EXECUTABLE=exe-name

#SBATCH --nodes=3
#SBATCH --tasks-per-node=8
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name=diffuse_example

# ============================================================================

echo "Starting run at: `date`"
srun ./${EXECUTABLE}
echo "Program finished with exit code $? at: `date`"
exit 0
