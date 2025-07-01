#!/bin/bash
h=${2:-120} # time expected in h
m=${3:-4}  # memory usage expetcted in G
sbatch  <<EOT
#!/bin/bash

# SLURM options:

#SBATCH --job-name=exec
#SBATCH --output=exec-%j.out
#SBATCH --partition=htc               # Partition choice (htc by default)

#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=${m}G             # Memory in MB per default
#SBATCH --time=${h}:00:00               # Max time limit = 7 days

#SBATCH --comment "Large commands to run wherever"
#SBATCH --licenses=sps                # Declaration of storage and/or software resources

echo "starting..." 
conda activate lnstr
echo "env set up..." 

export OMP_NUM_THREADS=1
date "+%b %d%t%H:%M"
echo ./$1
./$1
date "+%b %d%t%H:%M"
echo "Done!" 
EOT