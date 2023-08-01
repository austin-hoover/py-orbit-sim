"""Test {2, 2} Danilov distribution envelope matching."""
from __future__ import print_function

import numpy as np

from bunch import Bunch
from bunch import BunchTwissAnalysis
from orbit.envelope import DanilovEnvelope22
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccNode
from orbit.space_charge.sc2p5d import setSC2p5DAccNodes
from orbit.space_charge.envelope import DanilovEnvSolverNode
from orbit.space_charge.envelope import setDanilovEnvSolverNodes
from orbit.teapot import teapot
from orbit.utils import consts
import orbit_mpi
from spacecharge import DanilovEnvSolver22
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
    length=lattice_length,
    intensity=0.0,
)
envelope.match_bare(lattice, method="2D", solver_nodes=None)

# Turn on space charge --- add envelope solver nodes.
envelope.set_intensity(intensity)
env_solver_nodes = setDanilovEnvSolverNodes(
    lattice, 
    calc=DanilovEnvSolver22, 
    perveance=envelope.perveance, 
    max_sep=0.1, 
    min_sep=1.00e-06
)

# Save initial envelope parameters.
init_envelope_params = np.copy(envelope.params)

# Print the real-space moments after each period.
n_periods = 20
Sigmas_env = []
print("Tracking envelope")
print("period | <xx>      | <yy>      | <xy>")
print("--------------------------------------------")
for index in range(n_periods + 1):
    Sigma = envelope.cov()
    Sigma *= 1.00e+06
    Sigmas_env.append(Sigma)    
    print("{:>6} | {:.6f} | {:.6f} | {:.6f}".format(index, Sigma[0, 0], Sigma[2, 2], Sigma[0, 2]))
    
    envelope.track(lattice)

# PIC tracking

lattice = create_lattice(lattice_length, lattice_fill_fac, lattice_kq)
calc = SpaceChargeCalc2p5D(128, 128, 1)
path_length_min = 0.1
sc_nodes = setSC2p5DAccNodes(lattice, path_length_min, calc)

print("Generating bunch")
n_parts = int(1.00e+05)
envelope.params = init_envelope_params
bunch, params_dict = envelope.to_bunch(n_parts, no_env=True)
bunch_twiss_analysis = BunchTwissAnalysis()

print("Tracking bunch")
print("period | | <xx>      | <yy>      | <yy>       | <xx> diff  | <yy> diff  | <xy> diff ")
print("------------------------------------------------------------------------------------")
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
    Sigma_delta = Sigma - Sigmas_env[index]
    print(
        "{:>6} | {:>10.6f} | {:>10.6f} | {:>10.6f} | {:>10.6f} | {:>10.6f} | {:>10.6f}".format(
            index, 
            Sigma[0, 0], 
            Sigma[2, 2], 
            Sigma[0, 2],
            Sigma_delta[0, 0], 
            Sigma_delta[2, 2], 
            Sigma_delta[0, 2],
        )
    )
    
    lattice.trackBunch(bunch, params_dict)
