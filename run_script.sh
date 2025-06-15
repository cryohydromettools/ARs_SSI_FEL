#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J plot_ann
#SBATCH -p debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem-per-cpu=2100
#SBATCH --mail-user=christian010194@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -o plot_%j.out
#SBATCH -e plot_%j.err

#-----------------Toolchain---------------------------
# ----------------Modulos----------------------------
ml  Anaconda3/2020.02
eval "$(conda shell.bash hook)"
##conda activate cosipy_V1.4
conda activate MassBalanceMachine

srun python -u /home/cr2/cmtorres/repos/ARs_SSI_FEL/plot_EE_ARs_SSI_1999_ann.py 
