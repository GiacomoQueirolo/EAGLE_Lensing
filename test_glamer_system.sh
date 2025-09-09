# Systematical test for glamer to see what it produces

source ~/.bashrc

conda activate lnstr

WD=/pbs/home/g/gqueirolo/EAGLE/
cd $WD

mkdir logs_part

for i in $(seq 1 10)
do
    echo "Galaxy number " ${i}
    timestamp=$(date +%s)
    echo "started at " ${timestamp}
    ./slrmscp_python_medium.sh create_part_data.py -fn "particles_EAGLE_${timestamp}.csv" > ${WD}/logs_part/GalPart_${timestamp}.log
    cd ${WD}/glamer_scripts/test_system/
    
    # use the "standard" particle list, not the "short" one
    sbatch  --output large_exec-GalPart_${timestamp}.out  ./slrmscp_exec_large_TestSys.sh  "particles_EAGLE_${timestamp}.csv" 
    # doesn't need to wait for it to finish
    cd ${WD}
done

# now it must wait for all to finish
lines=4
while [ $lines -gt 0  ] ; 
    do 
        lines=`squeue -u gqueirolo | grep large_exec | wc -l ` 
        sleep 10
done

echo "all process stopped working"
for i in  ${WD}/logs_part/GalPart_*log
do
    more $i 
    out_i=${WD}/glamer_scripts/revamp_test/large_exec-`basename ${i/.log/.out}`
    more $out_i
done
