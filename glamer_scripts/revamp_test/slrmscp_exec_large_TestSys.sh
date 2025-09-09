#!/bin/bash

# SLURM options:

#SBATCH --job-name=large_exec_TestSys
#SBATCH --partition=htc_highmem        # Partition choice (htc by default)

#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=8G             # Memory in MB per default
#SBATCH --time=120:00:00             # Max time limit = 7 days

#SBATCH --mail-user=giacomo.queirolo@umontpellier.fr          # Where to send the e-mail
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --licenses=sps                # Declaration of storage and/or software resources

echo "starting..." 
#conda init
#conda activate lnstr
#echo "env set up..." 

export OMP_NUM_THREADS=1
date "+%b %d%t%H:%M"
echo "./build/revamp_test "
./build/revamp_test 
date "+%b%d%t%H:%M"
echo "Done!" 

