"""Test {2, 2} Danilov distribution envelope matching."""
from __future__ import print_function

import numpy as np

from bunch import Bunch
from bunch import BunchTwissAnalysis
from orbit.envelope import DanilovEnvelope22
from orbit.envelope import DanilovEnvelopeSolverNode22
from orbit.envelope import set_danilov_envelope_solver_nodes_22
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccNode
from orbit.space_charge.sc2p5d import setSC2p5DAccNodes
from orbit.teapot import teapot
from orbit.utils import consts
import orbit_mpi
from spacecharge import SpaceChargeCalc2p5D


def create_lattice(length=5.0, fill_fac=0.5, kq=0.5):
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
    
    max_length = 0.010
    for node in lattice.getNodes():
        if node.getLength() > max_length:
            node.setnParts(1 + int(node.getLength() / max_length))
            
    return lattice


# Create FODO lattice.
lattice_length = 5.0
lattice_fill_fac = 0.5
lattice_kq = 0.5
lattice = create_lattice(lattice_length, lattice_fill_fac, lattice_kq)

# Create {2, 2} Danilov distribution envelope.
mass = consts.mass_proton  # [GeV/c^2]
kin_energy = 1.0  # [GeV]
intensity = 1.00e+15
bunch_length = (45.0 / 64.0) * 248.0  # [m]
mode = 1  # {1, 2} determines sign of vorticity
eps_l = 20.0e-06  # nonzero intrinsic emittance [m * rad]
eps_x_frac = 0.5  # eps_x / eps_l

envelope = DanilovEnvelope22(
    eps_l=eps_l,
    mode=mode,
    eps_x_frac=eps_x_frac,
    mass=mass,
    kin_energy=kin_energy,
    length=bunch_length,
    intensity=intensity,
)

# Add envelope solver nodes.
path_length_min = 0.010
solver_nodes = set_danilov_envelope_solver_nodes_22(
    lattice,
    path_length_max=None,
    path_length_min=path_length_min,
    perveance=envelope.perveance,
)

# Match to the bare lattice.
envelope.match_bare(lattice, method="2D", solver_nodes=solver_nodes)

# Compute the matched envelope.
envelope.match(
    lattice, 
    solver_nodes=solver_nodes, 
    method="replace_avg", 
    tol=1.00e-04, 
    verbose=2
)

# Print the real-space moments after each period.
n_periods = 20
print("Tracking envelope")
print("period | <xx>      | <yy>      | <xy>")
print("--------------------------------------------")
for index in range(n_periods + 1):
    Sigma = envelope.cov()
    Sigma *= 1.00e+06
    print("{:>6} | {:.6f} | {:.6f} | {:.6f}".format(index, Sigma[0, 0], Sigma[2, 2], Sigma[0, 2]))
    
    envelope.track(lattice)