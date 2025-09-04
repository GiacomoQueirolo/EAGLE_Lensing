#!/bin/bash

echo "untested"
exit
# I do it in a mean way
sbatch slrmscp_python_short_but_heavy.sh $@
sbatch slrmscp_python_short_but_heavy_htc.sh $@
# consider the option to use more
#sbatch slrmscp_python_short_but_heavy_.sh $@
#sbatch slrmscp_python_short_but_heavy.sh $@

# repeat 
EL=0 # how many repetition
while [[ $EL -le 8 ]] 
do 
    # wait 3 mins
    sleep $((60*3))
    
    IDS=`squeue -u gqueirolo | tail -n +2 | awk '{print $1}'`
    TIM=`squeue -u gqueirolo | tail -n +2 | awk '{print $6}'`
    i=0
    kill_ids=[]
    for TM in ${TIM[@]}
    do 
        SEC=`echo ${TM[@]} | awk -F : '{print $2}'`
        MIN=`echo ${TM[@]} | awk -F : '{print $1}'`
        ((TMS=60*10#$MIN+10#$SEC))
        if [[ $TMS -eq 0 ]] 
            # this job has not started
            kill_ids[$i]=IDS[$i]
        i=$((i+1))
    done
    
    if [[ ${#kill_ids} -eq ${#IDS}]] 
    then
        echo "no job has started yet"
    else
        for kj in ${kill_ids}
        do 
            scancel ${kj}
        done
        EL=100 # exit the while loop    
    fi
done

if [[ $EL -eq 8 ]]
then 
    echo "no job has started yet! Something is wrong - killing all of them apart the 1st"
    for kj in ${kill_ids[@]:1}
        do 
            scancel ${kj}
    done
fi
    

