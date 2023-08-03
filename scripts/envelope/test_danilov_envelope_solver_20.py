"""Test KV distribution ({2, 0} Danilov distribution) envelope tracking."""
from __future__ import print_function

import numpy as np

from bunch import Bunch
from bunch import BunchTwissAnalysis
from orbit.envelope import DanilovEnvelope20
from orbit.envelope import DanilovEnvelopeSolverNode20
from orbit.envelope import set_danilov_envelope_solver_nodes_20
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

# Set envelope parameters.
mass = consts.mass_proton  # [GeV/c^2]
kin_energy = 1.0  # [GeV]
intensity = 1.0e+15
bunch_length = (45.0 / 64.0) * 248.0  # [m]
eps_x = 10.0e-06  # [mrad]
eps_y = 10.0e-06  # [mrad]

# Create envelope matched to bare lattice.
envelope = DanilovEnvelope20(
    eps_x=eps_x,
    eps_y=eps_y,
    mass=mass,
    kin_energy=kin_energy,
    length=bunch_length,
    intensity=intensity,
)

# Add envelope solver nodes.
path_length_min = 0.010
env_solver_nodes = set_danilov_envelope_solver_nodes_20(
    lattice, 
    path_length_min=path_length_min,
    perveance=envelope.perveance,
    eps_x=envelope.eps_x,
    eps_y=envelope.eps_y,
)

# Save initial envelope parameters.
init_envelope_params = np.copy(envelope.params)

# Print the real-space moments after each period.
n_periods = 20
sizes_env = []
print("Tracking envelope")
print("period | <xx>       | <yy>")
print("---------------------------------")
for index in range(n_periods + 1):
    cx, cxp, cy, cyp = envelope.params
    sig_x = 0.5 * cx
    sig_y = 0.5 * cy
    sig_x = sig_x * 1000.0
    sig_y = sig_y * 1000.0
    sizes_env.append([sig_x, sig_y])    
    print("{:>6} | {:>10.6f} | {:>10.6f}".format(index, sig_x, sig_y))
    
    envelope.track(lattice)
    
    
    
# PIC tracking

lattice = create_lattice(lattice_length, lattice_fill_fac, lattice_kq)
calc = SpaceChargeCalc2p5D(128, 128, 1)
sc_nodes = setSC2p5DAccNodes(lattice, path_length_min, calc)

print("Generating bunch")
envelope.set_params(init_envelope_params)
bunch, params_dict = envelope.to_bunch(n_parts=int(1e5), no_env=True)
bunch_twiss_analysis = BunchTwissAnalysis()
bunch_twiss_analysis.computeBunchMoments(bunch, 2, 0, 0)

print("Tracking bunch")
print("period | x_rms      | y_rms      | x_rms diff | y_rms diff")
print("-----------------------------------------------------------")
for index in range(n_periods + 1):
    order = 2
    dispersion_flag = False
    eps_norm_flag = False
    bunch_twiss_analysis.computeBunchMoments(bunch, order, dispersion_flag, eps_norm_flag)
    Sigma = np.zeros((4, 4))
    for i in range(4):
        for j in range(i + 1):
            Sigma[i, j] = Sigma[j, i] = bunch_twiss_analysis.getCorrelation(i, j)
    Sigma *= 1.00e+06
    sizes = [np.sqrt(Sigma[0, 0]), np.sqrt(Sigma[2, 2])]
    sizes_delta = np.subtract(sizes, sizes_env[index])
    print(
        "{:>6} | {:>10.6f} | {:>10.6f} | {:>10.6f} | {:>10.6f}".format(
            index, 
            sizes[0],
            sizes[1],
            sizes_delta[0],
            sizes_delta[1], 
        )
    )
    
    lattice.trackBunch(bunch, params_dict)
