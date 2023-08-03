"""Test KV distribution ({2, 0} Danilov distribution) envelope matching."""
from __future__ import print_function

import numpy as np

from bunch import Bunch
from envelope import DanilovEnvelopeSolver20
from orbit.envelope import DanilovEnvelope20
from orbit.envelope import DanilovEnvelopeSolverNode20
from orbit.envelope import set_danilov_envelope_solver_nodes_20
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccNode
from orbit.teapot import teapot
from orbit.utils import consts


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
    
    max_length = 0.01
    for node in lattice.getNodes():
        if node.getLength() > max_length:
            node.setnParts(1 + int(node.getLength() / max_length))
            
    return lattice


# Create FODO lattice.
lattice_length = 5.0
lattice_fill_fac = 0.5
lattice_kq = 0.5
lattice = create_lattice(lattice_length, lattice_fill_fac, lattice_kq)

# Create envelope.
mass = consts.mass_proton  # [GeV/c^2]
kin_energy = 1.0  # [GeV]
intensity = 1.0e+16
bunch_length = (45.0 / 64.0) * 248.0  # [m]
eps_x = 10.0e-06 # [mrad]
eps_y = 10.0e-06 # [mrad]
envelope = DanilovEnvelope20(
    eps_x=eps_x,
    eps_y=eps_y,
    mass=mass,
    kin_energy=kin_energy,
    length=bunch_length,
    intensity=intensity,
)

# Add envelope solver nodes.
solver_nodes = set_danilov_envelope_solver_nodes_20(
    lattice, 
    path_length_min=0.010,
    perveance=envelope.perveance,
    eps_x=envelope.eps_x,
    eps_y=envelope.eps_y,
)

# Match to the bare lattice.
envelope.match_bare(lattice, solver_nodes=solver_nodes)

# Match with space charge.
envelope.match_lsq_ramp_intensity(lattice, solver_nodes=solver_nodes, n_steps=15, verbose=2)

# Print the envelope size/slope after each period.
n_periods = 20
print("period | cx         | cxp        | cy         | cyp")
print("-----------------------------------------------------------")
for index in range(n_periods + 1):
    (cx, cxp, cy, cyp) = 1000.0 * envelope.params
    fstr = "{:>6} | {:>10.6f} | {:>10.6f} | {:>10.6f} | {:>10.6f}"
    print(fstr.format(index, cx, cxp, cy, cyp))
    envelope.track(lattice)