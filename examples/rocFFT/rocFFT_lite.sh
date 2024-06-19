#!/bin/bash
start=`date +%s`

# Get nrun and npernode from command line
while getopts "a:b:" opt
do
   case $opt in
      a ) nrun=$OPTARG ;;
      b ) npernode=$OPTARG ;;
      ? ) echo "unrecognized bash option $opt" ;; # Print helpFunction in case parameter is non-existent
   esac
done

cd ../../
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONPATH=$PYTHONPATH:$PWD/GPy/
export PYTHONWARNINGS=ignore

cd -

# name of your machine, processor model, number of compute nodes, number of cores per compute node, which are defined in .gptune/meta.json
declare -a machine_info=($(python -c "from gptune import *;
(machine, processor, nodes, cores)=list(GetMachineConfiguration());
print(machine, processor, nodes, cores)"))
machine=${machine_info[0]}
processor=${machine_info[1]}
nodes=${machine_info[2]}
cores=${machine_info[3]}

obj=time



database="gptune.db/rocFFT.json"  # the phrase rocFFT should match the application name defined in .gptune/meta.json
# rm -rf $database

# start the main loop
more=1
while [ $more -eq 1 ]
do

# call GPTune and ask for the next sample point
python ./rocFFT_kernel_tuning.py -nrun $nrun


# check whether GPTune needs more data
idx=$( jq -r --arg v0 $obj '.func_eval | map(.evaluation_result[$v0] == null) | index(true) ' $database )
if [ $idx = null ]
then
more=0
fi

# if so, call the application code
while [ ! $idx = null ]; 
do 
echo "idx $idx"    # idx indexes the record that has null objective function values
# write a large value to the database. This becomes useful in case the application crashes. 
bigval=1e30
jq --arg v0 $obj --argjson v1 $idx --argjson v2 $bigval '.func_eval[$v1].evaluation_result[$v0]=$v2' $database > tmp.json && mv tmp.json $database

declare -a input_para=($( jq -r --argjson v1 $idx '.func_eval[$v1].task_parameter' $database | jq -r '.[]'))
declare -a tuning_para=($( jq -r --argjson v1 $idx '.func_eval[$v1].tuning_parameter' $database | jq -r '.[]'))


# echo "input_para ${input_para[*]}"
# echo "tuning_para ${tuning_para[*]}"

#############################################################################
#############################################################################
# Modify the following according to your application !!! 


# get the task input parameters, the parameters should follow the sequence of definition in the python file
length=$(( input_para[0]  * 64 ))

# get the tuning parameters, the parameters should follow the sequence of definition in the python file
wgs=$((tuning_para[0] * 64))
tpt=$((tuning_para[1] * 64))
half_lds=$((tuning_para[2]))
direct_reg=$((tuning_para[3]))

# call the application
export OMP_NUM_THREADS=$(($cores / $npernode))

RUN_BIN="./rocfft_config_search"

echo "$RUN_BIN manual -l $length -b 1 -f 8 8 -w $wgs --tpt $tpt --half-lds $half_lds --direct-reg 1 | tee rocfft_kernel.log"
$RUN_BIN manual -l $length -b 1 -f 8 8 -w $wgs --tpt $tpt --half-lds $half_lds --direct-reg 1 | tee rocfft_kernel.log

# get the result (for this example: search the runlog)
result=$(grep ', ' kernel.log | sed 's/.*, //')

# write the data back to the database file
jq --arg v0 $obj --argjson v1 $idx --argjson v2 $result '.func_eval[$v1].evaluation_result[$v0]=$v2' $database > tmp.json && mv tmp.json $database
idx=$( jq -r --arg v0 $obj '.func_eval | map(.evaluation_result[$v0] == null) | index(true) ' $database )

#echo "--- end loop $idx"

#############################################################################
#############################################################################



done
done

end=`date +%s`

runtime=$((end-start))
echo "Total tuning time: $runtime"

