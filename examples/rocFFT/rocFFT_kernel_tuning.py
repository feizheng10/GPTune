#! /usr/bin/env python3

"""
Example of invocation of this script:
python heffte_RCI.py -npernode 1 -nrun 80

where:
	-npernode is the number of MPIs (in linear scale) per node for launching the application code
    -nrun is the number of calls per task
"""

################################################################################

import sys
import os
import numpy as np
import argparse
import pickle

# from callhybrid import GPTuneHybrid

sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))

from gptune import * # import all


from autotune.problem import *
from autotune.space import *
from autotune.search import *

# from callopentuner import OpenTuner
# from callhpbandster import HpBandSter
import math

################################################################################
def objectives(point):
	return min(point)

def cst1(px_i,py_i,px_o,py_o,nodes,npernode):
	pz_i_linear = int(nodes*npernode/2**(px_i+py_i))
	pz_o_linear = int(nodes*npernode/2**(px_o+py_o))
	px_i_linear = 2**(px_i)
	py_i_linear = 2**(py_i)
	px_o_linear = 2**(px_o)
	py_o_linear = 2**(py_o)
	# the first condition is input grid doesn't oversubscribe the nodes
	# the second condition is output grid doesn't oversubscribe the nodes
	# the third condition is input and output grids should have the same total MPI counts
	return pz_i_linear>0 and pz_o_linear>0 and px_i_linear*py_i_linear*pz_i_linear == px_o_linear*py_o_linear*pz_o_linear

def main():

	# Parse command line arguments

	# args   = parse_args()

	# Extract arguments
	npernode = 1 # args.npernode
	nrun = 1 # args.nrun
	TUNER_NAME = 'GPTune' # 'GPTuneHybrid'
	(machine, processor, nodes, cores) = GetMachineConfiguration()
	print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

	os.environ['MACHINE_NAME'] = machine
	os.environ['TUNER_NAME'] = TUNER_NAME

	nprocmax = nodes*cores

	# Task parameters
	# Note: kernel_type, length, precision, batch_size are potential task parameters 
	length = Integer(64, 128, base=64, name="length")

	# Input/tuning parameters
	factorization = Categorical()
	wgs = Integer(64, 256, base=64,name="wgs")
	tpt = Integer(64, 256, base=64,name="tpt")
	half_lds = Integer(0, 1, base=1,name="half_lds")
	direct_reg = Integer(0, 1, base=1,name="direct_reg")


	# Tuning Objective
	time   = Real(float("-Inf") , float("Inf"), name="time")

	IS = Space([length])
	PS = Space([factorization, wgs, tpt, half_lds, direct_reg])
	OS = Space([time])

	constraints = {"cst1" : cst1}
	models = {}
	constants={"nodes":nodes,"cores":cores,"npernode":npernode}

	""" Print all input and parameter samples """
	# print(IS, PS, OS, constraints, models)
	print('IS: \n', IS, '\nPS: \n',PS,'\nOS: \n',OS, '\nconstraints: \n',constraints,'\nmodels: \n',models)


	problem = TuningProblem(IS, PS, OS, objectives, constraints, None, constants=constants)
	computer = Computer(nodes = nodes, cores = cores, hosts = None)

	""" Set and validate options """
	options = Options()
	options['RCI_mode'] = True
	options['model_processes'] = 1
	# options['model_threads'] = 1
	options['model_restarts'] = 1
	# options['search_multitask_processes'] = 1
	# options['model_restart_processes'] = 1
	options['distributed_memory_parallelism'] = False
	options['shared_memory_parallelism'] = False
	options['model_class'] = 'Model_GPy_LCM' # 'Model_GPy_LCM'
	options['verbose'] = False
	# options['sample_class'] = 'SampleOpenTURNS'
	# options['search_algo'] = 'dual_annealing' #'maco' #'moead' #'nsga2' #'nspso'
	options['search_pop_size'] = 1000
	options['search_gen'] = 10
	options['search_more_samples'] = 1

	
	options.validate(computer = computer)

	giventask = [[128, 64, 128]]
	data = Data(problem)


	# # # the following makes sure the first sample is using default parameters
	# data.I = giventask
	# data.P = [[['4',128,20,10, 4, 0]]]


	if(TUNER_NAME=='GPTune'):
		gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))

		NI = len(giventask)
		NS = nrun
		(data, model, stats) = gt.MLA(NS=NS, NI=NI, Tgiven=giventask, NS1=max(NS//2, 1))
		# print("stats: ", stats)

		""" Print all input and parameter samples """
		for tid in range(NI):
			print("tid: %d"%(tid))
			print("    Transform sizes:%s %s %s"%(data.I[tid][0],data.I[tid][1],data.I[tid][2]))
			print("    Ps ", data.P[tid])
			print("    Os ", data.O[tid].tolist())
			print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))


if __name__ == "__main__":

	main()
