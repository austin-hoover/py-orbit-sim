"""Parallel optimization with ORBIT_MPI and scipy.

https://stackoverflow.com/questions/37159923/parallelize-a-function-call-with-mpi4py
"""
from scipy.optimize import minimize
import numpy as np

import orbit_mpi
from orbit_mpi import mpi_datatype


_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)


def function(x, stop):
    stop = orbit_mpi.MPI_Bcast(stop, mpi_datatype.MPI_INT, 0, _mpi_comm)    
    cost = 0.0
    if stop == 0:
        x = orbit_mpi.MPI_Bcast(x.tolist(), mpi_datatype.MPI_DOUBLE, 0, _mpi_comm) 
        cost_ = x[0]**4 - x[0]**2 + x[1]**2
        cost = orbit_mpi.MPI_Bcast(cost_, mpi_datatype.MPI_DOUBLE, 0, _mpi_comm)    
        if _mpi_rank == 0:
            print("cost={}".format(cost))
    return cost


if _mpi_rank == 0:
    stop = 0
    x0 = np.array([20.0, 0.0])
    x = minimize(function, x0, args=(stop))
    
    print("the argmin is " + str(x))
    stop = 1
    function(x0, stop)

else:
    stop = 0
    x0 = np.array([20.0, 0.0])
    while stop == 0:
        function(x0, stop)
