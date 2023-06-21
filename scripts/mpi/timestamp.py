from __future__ import print_function
import time

import orbit_mpi


_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

timestamp = time.strftime("%y%m%d%H%M%S")
if _mpi_rank == 1:
    timestamp = str(int(timestamp) + 1)
print("(rank {}) timestamp={}".format(_mpi_rank, timestamp))
    
timestamp = orbit_mpi.MPI_Bcast(timestamp, orbit_mpi.mpi_datatype.MPI_CHAR, 0, _mpi_comm)    
print("(rank {}) new timestamp={}".format(_mpi_rank, timestamp))