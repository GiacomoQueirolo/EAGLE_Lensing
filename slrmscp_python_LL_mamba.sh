#!/bin/bash

# SLURM options:

#SBATCH --job-name=LLmamba_job
#SBATCH --output=LLmamba_pythjob-%j.out
#SBATCH --partition=htc_highmem      # Partition choice (htc by default)

#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=4 #prev:8
#SBATCH --mem-per-cpu=32G #prev:4    # Memory in MB per default
#SBATCH --time=12:00:00               # Max time limit = 7 days

#SBATCH --mail-user=giacomo.queirolo@umontpellier.fr          # Where to send the e-mail
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --comment "test"
#SBATCH --licenses=sps                # Declaration of storage and/or software resources

source /pbs/home/g/gqueirolo/.bashrc
echo "starting..." 
#conda activate lnstr
mamba activate astro
echo "env set up..." 

export OMP_NUM_THREADS=1
date "+%b %d%t%H:%M"
echo python $@
python $@
date "+%b %d%t%H:%M"
echo "Done!" 

