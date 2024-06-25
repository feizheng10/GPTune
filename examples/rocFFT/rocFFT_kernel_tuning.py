#! /usr/bin/env python3

"""

"""

################################################################################

import sys
import os
import numpy as np
import argparse
import pickle
from itertools import permutations

# from callhybrid import GPTuneHybrid

sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))

from gptune import * # import all


from autotune.problem import *
from autotune.space import *
from autotune.search import *
from itertools import chain, combinations
from functools import reduce
from operator import mul


# from callopentuner import OpenTuner
# from callhpbandster import HpBandSter
import math

# not in use yet
supported_factors = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 17}

# Function to generate the power set of a given iterable
def power_set(iterable):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

# Function to return the set of supported threads per transform
def supported_threads_per_transform(factorization):
    tpts = set()
    tpt_candidates = power_set(factorization)
    for tpt in tpt_candidates:
        if not tpt:
            continue
        product = reduce(mul, tpt, 1)
        tpts.add(product)
    return tpts

# Generate all permutations of each factorization of n
def factorize(n):
    def factorize_recursive(x, start):
        result = []
        for i in range(start, int(x**0.5) + 1):
            if x % i == 0:
                sub_result = factorize_recursive(x // i, i)
                for item in sub_result:
                    result.append([i] + item)
                result.append([i, x // i])
        return result
    
    # Check if n is a prime number
    if n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1)):
        return [[n]]
    
    factorizations = factorize_recursive(n, 2)

    # Generate all permutations of each factorization
    all_permutations = set()
    for factors in factorizations:
        for perm in permutations(factors):
            all_permutations.add(perm)

    # Convert permutations to stacked integers connected with 0
    stacked_ints = set()
    for perm in all_permutations:
        stacked_int = ('0'.join(map(str, perm)) + '0')
        stacked_ints.add(stacked_int)

    return sorted(stacked_ints)

################################################################################
def objectives(point):
	return min(point)

def main():

	# Parse command line arguments

	args   = parse_args()

	# Extract arguments
	nrun = args.nrun
	TUNER_NAME = 'GPTune' # 'GPTuneHybrid'
	(machine, processor, nodes, cores) = GetMachineConfiguration()
	print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))

	os.environ['MACHINE_NAME'] = machine
	os.environ['TUNER_NAME'] = TUNER_NAME

	nprocmax = nodes*cores

	all_factorizations = factorize(64)
	#print("---------------------", all_factorizations)

	# Task parameters
	# Note: kernel_type, length, precision, batch_size are potential task parameters 
	length = Integer(1, 2, transform="normalize", name="length")
	#length = Categoricalnorm([64, 128], transform="onehot", name="length")

	# Input/tuning parameters

	wgs = Integer(1, 4, transform="normalize", name="wgs")
	tpt = Integer(1, 4, transform="normalize", name="tpt")
	half_lds = Categoricalnorm (['0', '1'], transform="onehot", name="half_lds")
	direct_reg = Categoricalnorm (['0', '1'], transform="onehot", name="direct_reg")
	factorization = Categoricalnorm(all_factorizations, transform="onehot", name="factorization")

	# Tuning Objective 
	time   = Real(float("-Inf") , float("Inf"), name="time")

	IS = Space([length])
	PS = Space([wgs, tpt, half_lds, direct_reg, factorization])
	OS = Space([time])

	# cst1 =
	cst2 = "( tpt < wgs )"
	cst3 = "( not half_lds or direct_reg )"
	constraints = {" cst2 " : cst2, " cst3 " : cst3 } # constraints for task

	models = {}
	constants={"nodes":nodes,"cores":cores}

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
	options['model_class'] = 'Model_GPy_LCM'
	options['verbose'] = False
	# options['sample_class'] = 'SampleOpenTURNS'
	# options['search_algo'] = 'dual_annealing' #'maco' #'moead' #'nsga2' #'nspso'
	options['search_pop_size'] = 1000
	options['search_gen'] = 10
	options['search_more_samples'] = 1

	
	options.validate(computer = computer)

	giventask = [[1]] # // for length 64
	data = Data(problem)


	# # # the following makes sure the first sample is using default parameters
	# data.I = giventask
	# data.P = [[['4',128,20,10, 4, 0]]]


	if(TUNER_NAME=='GPTune'):
		gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))

		NI = len(giventask)
		NS = nrun
		# (data, model, stats) = gt.SLA(NS=NS, Tgiven=giventask, NS1=max(NS//2, 1))
		(data, model, stats) = gt.MLA(NS=NS, NI=NI, Tgiven=giventask, NS1=max(NS//2, 1))
		# print("stats: ", stats)

		""" Print all input and parameter samples """
		for tid in range(NI):
			print("tid: %d"%(tid))
			print("    Transform sizes:%s"%(data.I[tid][0]))
			print("    Ps ", data.P[tid])
			print("    Os ", data.O[tid].tolist())
			print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

def parse_args():

	parser = argparse.ArgumentParser()

	# Algorithm related arguments
	parser.add_argument('-nrun', type=int, help='Number of runs per task')

	args   = parser.parse_args()
	return args

if __name__ == "__main__":

	main()
