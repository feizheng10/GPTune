#! /usr/bin/env python

# GPTune Copyright (c) 2019, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S.Dept. of Energy) and the University of
# California, Berkeley.  All rights reserved.
#
# If you have questions about your rights to use or distribute this software, 
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
#
# NOTICE. This Software was developed under funding from the U.S. Department 
# of Energy and the U.S. Government consequently retains certain rights.
# As such, the U.S. Government has been granted for itself and others acting
# on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
# the Software to reproduce, distribute copies to the public, prepare
# derivative works, and perform publicly and display publicly, and to permit
# other to do so.
#

import numpy as np
from problem import Problem
from data import Data
from typing import Collection, Callable
import mpi4py
from mpi4py import MPI
import os
import sys
import concurrent
from concurrent import futures	

from pathlib import Path
import importlib


class Computer(object):

    def __init__(self, nodes : int = 1, cores : int = 1, hosts : Collection = None):

        self.nodes = nodes
        self.cores = cores
        self.hosts = hosts
        if (hosts != None and nodes != len(hosts)):
            raise Exception('The number of elements in "hosts" does not match with the number of "nodes"')

    def evaluate_constraints(self, problem, point : Collection, inputs_only : bool = False, **kwargs):  # point is in the original spaces

#       kwargs['constraints_evaluation_parallelism']

        # points can be either a dict or a list of dicts on which to iterate

        cond = True
        for (cstname, cst) in problem.constraints.items():
            if (isinstance(cst, str)):
                try:
                    # {} has to be the global argument to eval
                    # and point the local one, otherwise,
                    # point will be corrupted / updated by eval
                    cond = eval(cst, {}, point)
                except Exception as inst:
                    if (inputs_only and isinstance(inst, NameError)):
                        pass
                    else:
                        raise Exception(f"Unexpected exception '{inst}' was raised while evaluating constraint '{cstname}'. Correct this constraint before calling the tuner again.")
            else:
                try:
                    kwargs2 = {}
                    sig = inspect.signature(cst)
                    for varname in point:
                        if (varname in sig.parameters):
                            kwargs2[varname] = point[varname]
                    cond = cst(**kwargs2)
                except Exception as inst:
                    if (isinstance(inst, TypeError)):
                        lst = inst.__str__().split()
                        if (len(lst) >= 5 and lst[1] == 'missing' and lst[3] == 'required' and lst[4] == 'positional'):
                            pass
                        else:
                            raise Exception(f"Unexpected exception '{inst}' was raised while evaluating constraint '{cstname}'. Correct this constraint before calling the tuner again.")
                    else:
                        raise Exception(f"Unexpected exception '{inst}' was raised while evaluating constraint '{cstname}'. Correct this constraint before calling the tuner again.")
            if (not cond):
                break

        return cond


    def evaluate_objective(self, problem : Problem, I : np.ndarray = None, P : Collection[np.ndarray] = None, options: dict=None):  # P and I are in the normalized space
        O = []
        for i in range(len(I)):
            t = I[i]
            I_orig = problem.IS.inverse_transform(np.array(t, ndmin=2))[0]		
            # kwargst = {problem.IS[k].name: I_orig[k] for k in range(problem.DI)}
            P2 = P[i]
            O2 = self.evaluate_objective_onetask(problem=problem, i_am_manager=True, I_orig=I_orig, P2=P2, options = options)
            tmp = np.array(O2).reshape((len(O2), problem.DO))
            O.append(tmp.astype(np.double))   #YL: convert single, double or int to double types

        return O

    # def evaluate_models_onepoint(self, problem : Problem, point: dict):  # a simple wrapper for problem.models, point is in the original space
    #     print('nani')
    #     O= problem.models(point)
    #     return O



    def evaluate_objective_onetask(self, problem : Problem, pids : Collection[int] = None, i_am_manager : bool = True, I_orig: Collection=None, P2 : np.ndarray = None, options:dict=None):  # P2 is in the normalized space

        if(problem.driverabspath is not None and options['distributed_memory_parallelism']):
            modulename = Path(problem.driverabspath).stem  # get the driver name excluding all directories and extensions
            sys.path.append(problem.driverabspath) # add path to sys
            module = importlib.import_module(modulename) # import driver name as a module 
            # func = getattr(module, funcName)
        else: 
            module =problem

        O2=[]
        kwargst = {problem.IS[k].name: I_orig[k] for k in range(problem.DI)}

        if (pids is None):
            pids = list(range(len(P2)))


        if (options['distributed_memory_parallelism'] and options['objective_evaluation_parallelism'] and i_am_manager):
            
            if(problem.driverabspath is None):
                raise Exception('objective_evaluation_parallelism and distributed_memory_parallelism require passing driverabspath to GPTune')

            nproc = min(options['objective_multisample_processes'],len(P2))
            mpi_comm = self.spawn(__file__, nproc, nthreads=1, kwargs=options) 
            kwargs_tmp = options
            if "mpi_comm" in kwargs_tmp:
                del kwargs_tmp["mpi_comm"]   # mpi_comm is not picklable
            _ = mpi_comm.bcast((self, problem,P2, I_orig, pids, kwargs_tmp), root=mpi4py.MPI.ROOT)

            tmpdata = mpi_comm.gather(None, root=mpi4py.MPI.ROOT)
            mpi_comm.Disconnect()

            # reordering is needed as tmpdata[p] stores p, p+nproc, p+2nproc, ... 
            Otmp=[]
            offset=[0] * (nproc+1)
            for p in range(int(nproc)):
                Otmp = Otmp + tmpdata[p]
                offset[p+1]=offset[p]+len(tmpdata[p])
            for it in range(len(tmpdata[0])): 
                for p in range(int(nproc)):
                    if(len(O2)<len(P2)):
                        O2.append(Otmp[offset[p]+it])
                                
        elif (options['shared_memory_parallelism'] and options['objective_evaluation_parallelism']):
            with concurrent.futures.ThreadPoolExecutor(max_workers = options['objective_multisample_threads']) as executor:
                def fun(pid):
                    x = P2[pid]
                    x_orig = problem.PS.inverse_transform(np.array(x, ndmin=2))[0]		
                    kwargs = {problem.PS[k].name: x_orig[k] for k in range(problem.DP)}
                    kwargs.update(kwargst)
                    # print(kwargs)
                    return module.objectives(kwargs)                    
                O2 = list(executor.map(fun, pids, timeout=None, chunksize=1))	
        else:

            for j in pids:
                x = P2[j]
                x_orig = problem.PS.inverse_transform(np.array(x, ndmin=2))[0]		
                kwargs = {problem.PS[k].name: x_orig[k] for k in range(problem.DP)}
                kwargs.update(kwargst)
                o = module.objectives(kwargs)
                # print('kwargs',kwargs,'o',o)
                O2.append(o)
        return O2



        

    def spawn(self, executable, nproc, nthreads, args=None, kwargs=None): 

        # XXX
#        check_mpi()
#        mpi_info = MPI.Info.Create()
#        mpi_info.Set("add-hostfile", "slurm.hosts")
#        mpi_info.Set("host", "slurm.hosts")
         

        print('exec', executable, 'args', args, 'nproc', nproc)#, info=mpi_info).Merge()# process_rank = comm.Get_rank()
#        comm = MPI.COMM_SELF.Spawn(executable, args=args, maxprocs=nproc)#, info=mpi_info).Merge()# process_rank = comm.Get_rank()
#        comm = MPI.COMM_SELF.Spawn('/usr/common/software/python/3.7-anaconda-2019.07/bin/python', args=executable, maxprocs=nproc)#, info=mpi_info).Merge()# process_rank = comm.Get_rank()
        
        info = MPI.Info.Create()
        info.Set('env', 'OMP_NUM_THREADS=%d\n' %(nthreads))        
        comm = MPI.COMM_SELF.Spawn(sys.executable, args=executable, maxprocs=nproc,info=info)#, info=mpi_info).Merge()# process_rank = comm.Get_rank()
        # process_rank = comm.Get_rank()
        # process_count = comm.Get_size()
        # process_host = MPI.Get_processor_name()
        # print('manager',process_rank, process_count, process_host)
        return comm

#print(MPI.COMM_WORLD.Get_rank(), MPI.Get_processor_name())



if __name__ == '__main__':

    def objectives(point):
        print('this is a dummy definition')
        return point
    def models(point):
        print('this is a dummy definition')
        return point        

    mpi_comm = MPI.Comm.Get_parent()
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    (computer, problem,P2, I_orig, pids, kwargs) = mpi_comm.bcast(None, root=0)
    pids_loc = pids[mpi_rank:len(pids):mpi_size]
    tmpdata = computer.evaluate_objective_onetask(problem, pids_loc, False, I_orig, P2, kwargs)
    res = mpi_comm.gather(tmpdata, root=0) 
    mpi_comm.Disconnect()	