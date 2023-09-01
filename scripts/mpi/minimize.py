"""Optimize drift length in quad-drift system."""
import time

import scipy.optimize
import numpy as np

from bunch import Bunch
from bunch import BunchTwissAnalysis
import orbit_mpi
from orbit.teapot import teapot


_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)


# Create lattice of quad + drift.
lattice = teapot.TEAPOT_Lattice()

quad_node = teapot.QuadTEAPOT()
quad_node.setParam("kq", 2.0)
lattice.addNode(quad_node)

drift_node = teapot.DriftTEAPOT()
drift_node.setLength(1.0)
lattice.addNode(drift_node)

lattice.initialize()

# Create a bunch.
bunch = Bunch()
bunch.mass(0.932)
bunch.charge(1.0)
bunch.getSyncParticle().kinEnergy(1.0)
n_parts = int(1.00e+02)
for i in range(n_parts):
    x = np.random.normal(scale=0.010)
    xp = np.random.normal(scale=0.010)
    bunch.addParticle(x, xp, 0.0, 0.0, 0.0, 0.0)


# Minimize beam size at the lattice exit by changing the drift length.

def cost_function(x):
    drift_node.setLength(x)
    bunch_in = Bunch()
    bunch.copyBunchTo(bunch_in)
    lattice.trackBunch(bunch_in)
    
    twiss_analysis = BunchTwissAnalysis()
    twiss_analysis.computeBunchMoments(bunch, 2, 0, 0)
    sig_xx = twiss_analysis.getCorrelation(0, 0)
    cost = np.sqrt(sig_xx)
    print("x={:0.3e} cost={:0.3e}".format(x[0], cost))
    return cost


x0 = np.array([5.0])
bounds = scipy.optimize.Bounds([0.0], [np.inf])

start_time = time.time()
result = scipy.optimize.minimize(
    cost_function, 
    x0,
    bounds=bounds,
    method="trust-constr",
    options=dict(verbose=2),
)
print(result)
print("runtime={}".format(time.time() - start_time))