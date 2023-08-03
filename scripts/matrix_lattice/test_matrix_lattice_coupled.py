"""Compute transfer matrix parameters."""
from __future__ import print_function
import os
import pathlib
from pprint import pprint 
import sys

import numpy as np
import pandas as pd

from bunch import Bunch 
from orbit.teapot import teapot
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_Ring
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice_Coupled
from orbit.utils import consts


def create_lattice(length=5.0, fill_fac=0.5, kq=0.5):
    """Return linear FODO lattice (one period).
    
    Parameters
    ----------
    length : float
        Lattice length [m].
    fill_fac : float
        Fraction of lattice filled with quadrupoles.
    kq : float
        Quadupole strength.
        
    Returns
    -------
    TEAPOT_Lattice
    """
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
    return lattice


# Create FODO lattice.
lattice_length = 5.0  # [m]
fill_fac = 0.5
kq = 0.5
lattice = create_lattice(length=lattice_length, fill_fac=fill_fac, kq=kq)

# Tilt a quadrupole.
quad_node = lattice.getNodeForName("qf")
quad_node.setTiltAngle(np.radians(0.0))

# Create empty bunch.
mass = consts.mass_proton  # particle mass [GeV / c^2]
kin_energy = 1.000 # synchronous particle energy [GeV]
bunch = Bunch()
bunch.mass(mass)
bunch.getSyncParticle().kinEnergy(kin_energy)

# Analyze the one-turn transfer matrix.
print()
print("Courant-Snyder parameters (MATRIX_Lattice):")
print("-------------------------------------------")
matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, bunch)
ring_params = matrix_lattice.getRingParametersDict()
pprint(ring_params)

print()
print("Lebedev-Bogacz parameters (MATRIX_Lattice_Coupled):")
print("----------------------------------------------------`")
matrix_lattice = TEAPOT_MATRIX_Lattice_Coupled(lattice, bunch, parameterization="LB")
ring_params = matrix_lattice.getRingParametersDict()
pprint(ring_params)

# Compute the s-dependent Twiss parameters.
#
# (This method goes to each node and computes the one-turn transfer
# matrix starting from that node (rather than tracking the Twiss 
# parameters around the ring). It has not been benchmarked.)
print()
print("Tracked Twiss parameters (MATRIX_Lattice_Coupled):")
print("--------------------------------------------------")
data = matrix_lattice.getRingTwissData()
df = pd.DataFrame()
for key in data:
    df[key] = data[key]
print(df)

## Save the s-dependent Twiss parameters.
# df.to_csv("twiss.dat")

# Track a particle.
eigenvectors = ring_params["eigenvectors"]
v1 = eigenvectors[:, 0]
v2 = eigenvectors[:, 2]
J1 = 10.00e-06
J2 = 0.00e-06
psi1 = np.radians(0.0)
psi2 = np.radians(0.0)
x = np.real(np.sqrt(2.0 * J1) * v1 * np.exp(-1.0j * psi1) + np.sqrt(2.0 * J2) * v2 * np.exp(-1.0j * psi2))

bunch.deleteAllParticles()
bunch.compress()
bunch.addParticle(x[0], x[1], x[2], x[3], 0.0, 0.0)

print()
print("Tracking particle")
print("period | x        | x'       | y        | y'")
print("--------------------------------------------------")
n_periods = 20
for index in range(n_periods + 1):
    scale = 1000.0
    x = scale * bunch.x(0)
    y = scale * bunch.y(0)
    xp = scale * bunch.xp(0)
    yp = scale * bunch.yp(0)
    print(
        "{:>6} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f}".format(
            index, x, xp, y, yp,
        )
    )
    lattice.trackBunch(bunch)