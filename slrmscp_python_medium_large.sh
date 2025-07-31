#!/bin/bash

# SLURM options:

#SBATCH --job-name=medLarge_job
#SBATCH --output=medLarge_pythjob-%j.out
#SBATCH --partition=htc_highmem      # Partition choice (htc by default)

#SBATCH --ntasks=1                    # Run a single task
#SBATCH --mem=12G                    # Memory in MB per default
#SBATCH --time=1:00:00             # Max time limit = 7 days # #SBATCH --mem=4G                    # Memory in MB per default #SBATCH --time=04:00:00             # Max time limit = 7 days

#SBATCH --mail-user=giacomo.queirolo@umontpellier.fr          # Where to send the e-mail
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --comment "Medium python commands to run wherever"
#SBATCH --licenses=sps                # Declaration of storage and/or software resources

echo "starting..." 
source activate stenv
echo "env set up..." 

export OMP_NUM_THREADS=1
date "+%b %d%t%H:%M"
echo python $@
python $@
date "+%b%d%t%H:%M"
echo "Done!" 

