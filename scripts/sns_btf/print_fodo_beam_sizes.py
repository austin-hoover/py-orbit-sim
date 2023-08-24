from __future__ import print_function
import math
import os
import pathlib
import pickle
from pprint import pprint
import shutil
import sys
import time

import numpy as np

from bunch import Bunch
from bunch import BunchTwissAnalysis
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.utils import consts
import orbit_mpi

from sns_btf import SNS_BTF

import pyorbit_sim


_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)


file_dir = os.path.dirname(os.path.realpath(__file__))  # directory of this file
input_dir = os.path.join(file_dir, "data_input")  # lattice input data


# Lattice
# ------------------------------------------------------------------------------

# Settings
xml_filename = os.path.join(input_dir, "xml/btf_lattice_straight.xml")
coef_filename = os.path.join(input_dir, "magnets/default_i2gl_coeff_straight.csv")
mstate_filename = None
quads_filename = os.path.join(
    "/home/46h/sim_data/SNS_BTF/match_parallel/2023-08-09/",
    "230809005942-match_parallel_quad_strengths_001575.dat"
)
sequences = [
    "MEBT1",
    "MEBT2",
]
max_drift_length = 0.010  # [m]


# Initialize
linac = SNS_BTF(coef_filename=coef_filename, rf_frequency=402.5e06)
linac.init_lattice(
    xml_filename=xml_filename,
    sequences=sequences,
    max_drift_length=max_drift_length,
)
if mstate_filename is not None:
    linac.update_quads_from_mstate(filename=mstate_filename, value_type="current")
if quads_filename is not None:
    linac.set_quads_from_file(quads_filename, comment="#", verbose=True)
linac.set_overlapping_pmq_fields(z_step=max_drift_length, verbose=True)
linac.add_uniform_ellipsoid_space_charge_nodes(
    n_ellipsoids=1,
    path_length_min=max_drift_length,
)
linac.add_aperture_nodes(drift_step=0.1, verbose=True)
linac.set_linac_tracker(True)
lattice = linac.lattice



# Bunch
# ------------------------------------------------------------------------------

# Settings
filename = os.path.join(
    "/home/46h/projects/BTF/sim/SNS_RFQ/parmteq/2021-01-01_benchmark/data/",
    "bunch_RFQ_output_1.00e+05.dat"
)
mass = 0.939294  # [GeV / c^2]
charge = -1.0  # [elementary charge units]
kin_energy = 0.0025  # [GeV]
current = 0.042  # [A]
n_parts = int(1.00e+04)  # max number of particles
intensity = pyorbit_sim.bunch_utils.get_intensity(current, linac.rf_frequency)

# Initialize
bunch = Bunch()
bunch.mass(mass)
bunch.charge(charge)
bunch.getSyncParticle().kinEnergy(kin_energy)
bunch = pyorbit_sim.bunch_utils.load(
    filename=filename,
    bunch=bunch,
    verbose=True,
)
bunch = pyorbit_sim.bunch_utils.set_centroid(bunch, centroid=0.0)
bunch = pyorbit_sim.bunch_utils.downsample(
    bunch,
    n=n_parts,
    method="first",
    conserve_intensity=True,
    verbose=True,
)
bunch_size_global = bunch.getSizeGlobal()
macro_size = intensity / bunch_size_global
bunch.macroSize(macro_size)


# Tracking
# --------------------------------------------------------------------------------------

# Perturb the lattice.
np.random.seed(0)
node_names = ["MEBT:QV09", "MEBT:QH10"]
max_frac_delta = 0.1
for name in node_names:
    node = lattice.getNodeForName(name)
    kappa = node.getParam("dB/dr")
    # delta = kappa * np.random.uniform(-max_frac_delta, max_frac_delta) 
    delta = 0.2 * kappa
    node.setParam("dB/dr", kappa + delta)


monitor = pyorbit_sim.linac.BeamSizeMonitorFast(
    node_names=linac.quad_names_fodo[1::2],
    verbose=True
)
action_container = AccActionsContainer()
action_container.addAction(monitor.action, AccActionsContainer.ENTRANCE)

lattice.trackBunch(
    bunch,
    index_start=0,
    index_stop=-1,
    actionContainer=action_container,
)
