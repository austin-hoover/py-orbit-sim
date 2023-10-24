from __future__ import print_function

import numpy as np

import orbit_mpi
from bunch import Bunch
from spacecharge import Grid2D

import pyorbit_sim


# MPI
_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

# Create bunch.
n = int(1.00e+06)
n_local = n // _mpi_size
xmax = 5.0
bunch = Bunch()
bunch.macroSize(1.0)
for i in range(n_local):
    x, xp, y, yp, z, dE = np.random.uniform(-xmax, xmax, size=6)
    bunch.addParticle(x, xp, y, yp, z, dE)
print("(rank {}) bunch.getSize() = {}".format(_mpi_rank, bunch.getSize()))

# Define grid.
axis = (0, 2)
n_bins = 2 * [5]
limits = None
histogram = pyorbit_sim.diagnostics.Histogram2D(
    axis=axis,
    n_bins=n_bins,
    limits=limits,
)

histogram.action(bunch)

if _mpi_rank == 0:
    hist, edges = histogram.get_hist()
    print(_mpi_rank, histogram.hist.shape)
    print(_mpi_rank, [e.shape for e in histogram.edges])
    print(hist)
