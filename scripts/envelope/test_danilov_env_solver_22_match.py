"""Test {2, 2} Danilov distribution envelope matching."""
from __future__ import print_function

import numpy as np

from bunch import Bunch
from orbit.envelope import DanilovEnvelope22
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccNode
from orbit.space_charge.envelope import DanilovEnvSolverNode
from orbit.space_charge.envelope import setDanilovEnvSolverNodes
from orbit.teapot import teapot
from orbit.utils import consts
from spacecharge import DanilovEnvSolver22


# Create FODO lattice.
length = 5.0
fill_fac = 0.5
kq = 0.5

lattice = teapot.TEAPOT_Lattice()

node = teapot.DriftTEAPOT(name="drift1")
node.setLength(length * 0.25 * fill_fac)
lattice.addNode(node)

node = teapot.QuadTEAPOT(name="qf")
node.setLength(length * 0.5 * fill_fac)
node.addParam("kq", +kq)
lattice.addNode(node)

node = teapot.DriftTEAPOT(name="drift2")
node.setLength(length * 0.5 * fill_fac)
lattice.addNode(node)

node = teapot.QuadTEAPOT(name="qd")
node.setLength(length * 0.5 * fill_fac)
node.addParam("kq", -kq)
lattice.addNode(node)

node = teapot.DriftTEAPOT(name="drift3")
node.setLength(length * 0.25 * fill_fac)
lattice.addNode(node)

for node in lattice.getNodes():
    node.setUsageFringeFieldIN(False)
    node.setUsageFringeFieldOUT(False)
lattice.initialize()

# Set envelope parameters.
mass = consts.mass_proton  # [GeV/c^2]
kin_energy = 1.0  # [GeV]
intensity = 0.5e+14
bunch_length = (45.0 / 64.0) * 248.0  # [m]
mode = 1  # {1, 2} determines sign of vorticity
eps_l = 20.0e-06  # nonzero intrinsic emittance [m * rad]
eps_x_frac = 0.5  # eps_x / eps_l

# Create envelope matched to bare lattice.
envelope = DanilovEnvelope22(
    eps_l=eps_l,
    mode=mode,
    eps_x_frac=eps_x_frac,
    mass=mass,
    kin_energy=kin_energy,
    length=length,
    intensity=0.0,
)
envelope.match_bare(lattice, method="2D", solver_nodes=None)

# Match to the lattice with space charge.
envelope.set_intensity(intensity)
env_solver_nodes = setDanilovEnvSolverNodes(
    lattice, 
    calc=DanilovEnvSolver22, 
    perveance=envelope.perveance, 
    max_sep=0.1, 
    min_sep=1.00e-06
)
envelope.match(
    lattice,
    solver_nodes=env_solver_nodes, 
    method="lsq", 
    tol=1.00e-04,
    verbose=2
)

# Print the real-space moments after each period.
n_periods = 20
print("period | <xx>      | <yy>      | <xy>")
print("--------------------------------------------")
for index in range(n_periods + 1):
    envelope.track(lattice)
    Sigma = envelope.cov()
    Sigma *= 1.00e+06
    print("{:>6} | {:.6f} | {:.6f} | {:.6f}".format(index, Sigma[0, 0], Sigma[2, 2], Sigma[0, 2]))