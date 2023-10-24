from __future__ import print_function
import random

import orbit_mpi
from bunch import Bunch
from spacecharge import Grid2D


# MPI
_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)


# Define grid.
xmax = 4.0
n_bins = 5
grid = Grid2D(n_bins, n_bins, -xmax, xmax, -xmax, xmax)


# Create bunch.
n = 1000
n_local = n // _mpi_size
bunch = Bunch()
bunch.macroSize(1.0)
for i in range(n_local):
    x = random.uniform(-xmax, xmax)
    y = random.uniform(-xmax, xmax)
    bunch.addParticle(x, 0.0, y, 0.0, 0.0, 0.0)
print("(rank {}) bunch.getSize() = {}".format(_mpi_rank, bunch.getSize()))

# Compute histogram.
axis = (0, 2)
grid.binBunch(bunch, axis[0], axis[1])
grid.synchronizeMPI(_mpi_comm)
for i in range(n_bins):
    for j in range(n_bins):
        value = grid.getValueOnGrid(i, j)
        print("(rank {}) grid[{}, {}] = {:0.3f}".format(_mpi_rank, i, j, value))
        