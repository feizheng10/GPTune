#!/bin/bash


##################################################
##################################################

BuildExample=0 # whether all the examples have been built


# ################ Any mac os machine that has used config_macbook.sh to build GPTune
export machine=macbook
export proc=inteli7   
export mpi=openmpi  
export compiler=gnu   
export nodes=1  # number of nodes to be used

# ################ Cori
# export machine=cori
# export proc=haswell   # knl,haswell
# export mpi=craympich  # openmpi,craympich
# export compiler=gnu   # gnu, intel	
# export nodes=1  # number of nodes to be used


# ################ Yang's tr4 machine
# export machine=yang
# export proc=tr4   
# export mpi=openmpi  
# export compiler=gnu   
# export nodes=1  # number of nodes to be used


# ################ Any ubuntu machine that has used config_cleanubuntu.sh to build GPTune
# export machine=cleanubuntu
# export proc=unknown   
# export mpi=openmpi  
# export compiler=gnu   
# export nodes=1  # number of nodes to be used


##################################################
##################################################



export ModuleEnv=$machine-$proc-$mpi-$compiler
############### Yang's tr4 machine
if [ $ModuleEnv = 'yang-tr4-openmpi-gnu' ]; then
    module load gcc/9.1.0
    module load openmpi/gcc-9.1.0/4.0.1
    module load scalapack-netlib/gcc-9.1.0/2.0.2
    module load python/gcc-9.1.0/3.7.4
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    MPIRUN=mpirun
    cores=16
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,0,2]},\"gcc\":{\"version_split\": [9,1,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,0,2]},\"gcc\":{\"version_split\": [9,1,0]}}")
# fi
###############
############### macbook
elif [ $ModuleEnv = 'macbook-inteli7-openmpi-gnu' ]; then
    export PYTHONPATH=$PYTHONPATH:$PWD/pygmo2/build/
	export PATH=/usr/local/Cellar/python@3.9/$pythonversion/bin/:$PATH
	export PATH=$PWD/env/bin/:$PATH
	export BLAS_LIB=/usr/local/Cellar/openblas/$openblasversion/lib/libblas.dylib
	export LAPACK_LIB=/usr/local/Cellar/lapack/$lapackversion/lib/liblapack.dylib

	export MPIRUN="$PWD/openmpi-4.0.1/bin/mpirun"
	export PATH=$PATH:$PWD/openmpi-4.0.1/bin
	export SCALAPACK_LIB=$PWD/scalapack-2.1.0/build/install/lib/libscalapack.dylib
	export LD_LIBRARY_PATH=$PWD/scalapack-2.1.0/build/install/lib/:$LD_LIBRARY_PATH
    export LIBRARY_PATH=$PWD/scalapack-2.1.0/build/install/lib/:$LIBRARY_PATH
    export DYLD_LIBRARY_PATH=$PWD/scalapack-2.1.0/build/install/lib/:$DYLD_LIBRARY_PATH

	export LD_LIBRARY_PATH=$PWD/openmpi-4.0.1/lib:$LD_LIBRARY_PATH
	export LIBRARY_PATH=$PWD/openmpi-4.0.1/lib:$LIBRARY_PATH  
	export DYLD_LIBRARY_PATH=$PWD/openmpi-4.0.1/lib/:$DYLD_LIBRARY_PATH
	
    export LD_LIBRARY_PATH=$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/:$LD_LIBRARY_PATH
    export LIBRARY_PATH=$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/:$LIBRARY_PATH
    export DYLD_LIBRARY_PATH=$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/:$DYLD_LIBRARY_PATH
    cores=8
    
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [10,2,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [10,2,0]}}")
# fi


############### Cori Haswell CrayMPICH+GNU
elif [ $ModuleEnv = 'cori-haswell-craympich-gnu' ]; then
    module load python/3.7-anaconda-2019.10
    module swap PrgEnv-intel PrgEnv-gnu
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
    MPIRUN=mpirun
    cores=32
    software_json=$(echo ",\"software_configuration\":{\"cray-mpich\":{\"version_split\": [7,7,10]},\"libsci\":{\"version_split\": [19,6,1]},\"gcc\":{\"version_split\": [8,3,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"cray-mpich\":{\"version_split\": [7,7,10]},\"libsci\":{\"version_split\": [19,6,1]},\"gcc\":{\"version_split\": [8,3,0]}}")
# fi
###############

############### Cori Haswell CrayMPICH+Intel
elif [ $ModuleEnv = 'cori-haswell-craympich-intel' ]; then
    module load python/3.7-anaconda-2019.10
    module swap PrgEnv-gnu PrgEnv-intel 
    module swap intel intel/19.0.3.199 
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
    MPIRUN=mpirun
    cores=32
    software_json=$(echo ",\"software_configuration\":{\"cray-mpich\":{\"version_split\": [7,7,10]},\"libsci\":{\"version_split\": [19,6,1]},\"intel\":{\"version_split\": [19,0,3]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"cray-mpich\":{\"version_split\": [7,7,10]},\"libsci\":{\"version_split\": [19,6,1]},\"intel\":{\"version_split\": [19,0,3]}}")    
# fi
###############



############### Cori Haswell Openmpi+GNU
elif [ $ModuleEnv = 'cori-haswell-openmpi-gnu' ]; then
    module load python/3.7-anaconda-2019.10
    module unload cray-mpich
    module swap PrgEnv-intel PrgEnv-gnu
    module load openmpi/4.0.1
    export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
    MPIRUN=mpirun
    cores=32
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")
# fi    
###############

############### Cori Haswell Openmpi+Intel
elif [ $ModuleEnv = 'cori-haswell-openmpi-intel' ]; then
    module load python/3.7-anaconda-2019.10
    module unload cray-mpich
    module swap PrgEnv-gnu PrgEnv-intel 
    module swap intel intel/19.0.3.199 
    module load openmpi/4.0.1
    export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
    MPIRUN=mpirun
    cores=32
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"intel\":{\"version_split\": [19,0,3]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"intel\":{\"version_split\": [19,0,3]}}")    
# fi    
###############


############### Cori KNL Openmpi+GNU
elif [ $ModuleEnv = 'cori-knl-openmpi-gnu' ]; then
	module unload darshan
	module swap craype-haswell craype-mic-knl
	module load craype-hugepages2M
	module unload cray-libsci
	module unload cray-mpich
	module swap PrgEnv-intel PrgEnv-gnu
	module load openmpi/4.0.1
    export OMPI_MCA_btl_ugni_virtual_device_count=1
    export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
    MPIRUN=mpirun
    cores=64
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")    
# fi    
###############


############### Cori KNL Openmpi+Intel
elif [ $ModuleEnv = 'cori-knl-openmpi-intel' ]; then
	module unload darshan
	module swap craype-haswell craype-mic-knl
	module load craype-hugepages2M
	module unload cray-libsci
	module unload cray-mpich
	module swap PrgEnv-gnu PrgEnv-intel 
    module swap intel intel/19.0.3.199 
	module load openmpi/4.0.1
    MPIRUN=mpirun
    export OMPI_MCA_btl_ugni_virtual_device_count=1
    export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
    cores=64
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"intel\":{\"version_split\": [19,0,3]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"intel\":{\"version_split\": [19,0,3]}}")       
elif [ $ModuleEnv = 'cleanubuntu-unknown-openmpi-gnu' ]; then

	export PATH=$PWD/env/bin/:$PATH
	export PATH=$PATH:$PWD/openmpi-4.0.1/bin
	export MPIRUN=$PWD/openmpi-4.0.1/bin/mpirun
	export LD_LIBRARY_PATH=$PWD/scalapack-2.1.0/build/install/lib/:$LD_LIBRARY_PATH
	export LD_LIBRARY_PATH=$PWD/openmpi-4.0.1/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/

    cores=4
    
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,4,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,4,0]}}")
else
    echo "Untested ModuleEnv: $ModuleEnv, please add the corresponding definitions in this file"
    exit
fi    
###############

export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONPATH=$PYTHONPATH:$PWD/GPy/
export PYTHONWARNINGS=ignore
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/STRUMPACK/STRUMPACK/install/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/ButterflyPACK/ButterflyPACK/build/lib   # needed by strumpack_MLA_KRR.py
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/ButterflyPACK/ButterflyPACK/arpack-ng/build/lib   # needed by strumpack_MLA_KRR.py
export LD_LIBRARY_PATH=$PWD/scalapack-2.1.0/build/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/examples/STRUMPACK/STRUMPACK/install/include/python/:$PYTHONPATH
export GPTUNEROOT=$PWD




machine_json=$(echo ",\"machine_configuration\":{\"machine_name\":\"$machine\",\"$proc\":{\"nodes\":$nodes,\"cores\":$cores}}")
loadable_machine_json=$(echo ",\"loadable_machine_configurations\":{\"$machine\":{\"$proc\":{\"nodes\":$nodes,\"cores\":$cores}}}")









##########################################################################
##########################################################################
echo "Testing MPI Spawning-based Interface"

# the following examples only work with openmpi
if [[ $ModuleEnv == *"openmpi"* ]]; then

    cd $GPTUNEROOT/examples/GPTune-Demo
    rm -rf gptune.db/*.json # do not load any database 
    tp=GPTune-Demo
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
    $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./demo.py

    cd $GPTUNEROOT/examples/Scalapack-PDGEQRF
    rm -rf gptune.db/*.json # do not load any database 
    tp=PDGEQRF
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
    $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./scalapack_MLA.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -nprocmin_pernode 1 -ntask 2 -nrun 40 -machine cori -jobid 0 -tla 0
    $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./scalapack_MLA.py -mmax 1000 -nmax 1000 -nodes 1 -cores 4 -nprocmin_pernode 1 -ntask 2 -nrun 20 -machine cori -jobid 0 -tla 1

    if [[ $BuildExample == 1 ]]; then
        cd $GPTUNEROOT/examples/SuperLU_DIST
        rm -rf gptune.db/*.json # do not load any database 
        tp=SuperLU_DIST
        app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
        echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
        $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./superlu_MLA.py  -nodes 1 -cores 4 -nprocmin_pernode 1 -ntask 1 -nrun 20 -machine cori


        cd $GPTUNEROOT/examples/STRUMPACK
        rm -rf gptune.db/*.json # do not load any database 
        tp=STRUMPACK_Poisson3d
        app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
        echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
        $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./strumpack_MLA_Poisson3d.py  -nodes 1 -cores 4 -ntask 1 -nrun 10 -machine cori 


        ###### this one has a segmentation fault when running on Cori, also seems to fail on MacOS
        cd $GPTUNEROOT/examples/STRUMPACK
        rm -rf gptune.db/*.json # do not load any database
        tp=STRUMPACK_KRR
        app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
        echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json 
        $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./strumpack_MLA_KRR.py  -nodes 1 -cores 4 -ntask 1 -nrun 10 -machine cori 



        cd $GPTUNEROOT/examples/MFEM
        rm -rf gptune.db/*.json # do not load any database 
        tp=MFEM
        app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
        echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
        $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./mfem_maxwell3d.py -ntask 1 -nrun 20 -nprocmin_pernode 2 -optimization GPTune


        cd $GPTUNEROOT/examples/ButterflyPACK
        rm -rf gptune.db/*.json # do not load any database 
        tp=ButterflyPACK-IE2D
        app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
        echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
        $MPIRUN --oversubscribe --allow-run-as-root --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1  python ./butterflypack_ie2d.py -ntask 1 -nrun 20 -machine tr4 -nprocmin_pernode 2 -optimization GPTune 
    fi

fi




##########################################################################
##########################################################################
echo "Testing Reverse Communication Interface"


cd $GPTUNEROOT/examples/Scalapack-PDGEQRF_RCI
rm -rf gptune.db/*.json # do not load any database 
tp=PDGEQRF
app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
bash scalapack_MLA_RCI.sh -a 10 -b 2  #a: nrun b: nprocmin_pernode


if [[ $BuildExample == 1 ]]; then
    cd $GPTUNEROOT/examples/SuperLU_DIST_RCI
    rm -rf gptune.db/*.json # do not load any database 
    tp=SuperLU_DIST
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
    bash superlu_MLA_RCI.sh -a 10 -b 2 -c memory #a: nrun b: nprocmin_pernode c: objective

    cd $GPTUNEROOT/examples/SuperLU_DIST_RCI
    rm -rf gptune.db/*.json # do not load any database 
    tp=SuperLU_DIST
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
    bash superlu_MLA_MO_RCI.sh -a 10 -b 2 #a: nrun b: nprocmin_pernode 


    cd $GPTUNEROOT/examples/MFEM_RCI
    rm -rf gptune.db/*.json # do not load any database 
    tp=MFEM
    app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
    echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json
    bash mfem_maxwell3d_RCI.sh -a 20 -b 2  #a: nrun b: nprocmin_pernode
fi

##########################################################################
##########################################################################