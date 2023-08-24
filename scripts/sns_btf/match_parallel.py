"""Match the beam to the FODO line."""
from __future__ import print_function
import os
import pathlib
from pprint import pprint
import sys
import time

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

from bunch import Bunch
from bunch import BunchTwissAnalysis
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.py_linac.lattice import LinacAccLattice
from orbit.py_linac.lattice import OverlappingQuadsNode
from orbit.py_linac.lattice import Quad
import orbit_mpi
from orbit_mpi import mpi_datatype
from orbit_mpi import mpi_op

from sns_btf import BeamSizeMonitor
from sns_btf import Matcher
from sns_btf import OpticsController
from sns_btf import SNS_BTF

import pyorbit_sim


# Setup
# --------------------------------------------------------------------------------------

save = True

# MPI
_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

# Broadcast timestamp from MPI rank 0.
main_rank = 0
datestamp = time.strftime("%Y-%m-%d")
timestamp = time.strftime("%y%m%d%H%M%S")
datestamp = orbit_mpi.MPI_Bcast(datestamp, orbit_mpi.mpi_datatype.MPI_CHAR, main_rank, _mpi_comm)
timestamp = orbit_mpi.MPI_Bcast(timestamp, orbit_mpi.mpi_datatype.MPI_CHAR, main_rank, _mpi_comm)

# Set up directories.
file_dir = os.path.dirname(os.path.realpath(__file__))
input_dir = os.path.join(file_dir, "data_input")
output_dir = "/home/46h/repo/py-orbit-sim/data_output/"
if _mpi_rank == 0:
    print("file_dir = {}".format(file_dir))
    print("input_dir = {}".format(input_dir))
    print("output_dir = {}".format(output_dir))

# Create output directory and save script info.
man = pyorbit_sim.utils.ScriptManager(
    datadir=output_dir,
    path=pathlib.Path(__file__),
    timestamp=timestamp,
    datestamp=datestamp,
    script_path_in_outdir=True,
)
if save and _mpi_rank == 0:
    man.make_outdir()
    man.save_info()
    man.save_script_copy()
    pprint(man.get_info())


# Lattice
# ------------------------------------------------------------------------------

# Settings
xml_filename = os.path.join(input_dir, "xml/btf_lattice_straight.xml")
coef_filename = os.path.join(input_dir, "magnets/default_i2gl_coeff_straight.csv")
mstate_filename = None
quads_filename = os.path.join(
    "/home/46h/sim_data/SNS_BTF/match_parallel/2023-08-08/",
    "230808225233-match_parallel_quad_strengths_001750.dat"
)
sequences = [
    "MEBT1",
    "MEBT2",
]
max_drift_length = 0.010  # [m]

# Initialize lattice.
linac = SNS_BTF(coef_filename=coef_filename, rf_frequency=402.5e+06)
linac.init_lattice(
    xml_filename=xml_filename,
    sequences=sequences,
    max_drift_length=max_drift_length,
)
linac.set_overlapping_pmq_fields(z_step=max_drift_length, verbose=True)
linac.add_uniform_ellipsoid_space_charge_nodes(
    n_ellipsoids=3,
    path_length_min=max_drift_length,
)
linac.set_linac_tracker(True)
lattice = linac.lattice


# Bunch
# ------------------------------------------------------------------------------

# Settings
filename = os.path.join(
    "/home/46h/projects/BTF/sim/SNS_RFQ/parmteq/2021-01-01_benchmark/data/",
    "bunch_RFQ_output_1.00e+05.dat",
)
mass = 0.939294  # [GeV / c^2]
charge = -1.0  # [elementary charge units]
kin_energy = 0.0025  # [GeV]
current = 0.042  # [A]
n_parts = int(1.00e+04)  # max number of particles
intensity = pyorbit_sim.bunch_utils.get_intensity(current, linac.rf_frequency)

# Initialize the bunch.
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
)
bunch_size_global = bunch.getSizeGlobal()
macro_size = intensity / bunch_size_global
bunch.macroSize(macro_size)


# Matching to FODO channel
# ------------------------------------------------------------------------------

if _mpi_rank == 0:
    print("Matching to FODO channel")


# Update start/stop node indices.
index_start = 0
index_stop = lattice.getNodeIndex(lattice.getNodeForName("MEBT:QH30"))

# Identify FODO quads.
fodo_quad_names = linac.quad_names_fodo
if _mpi_rank == 0:
    print("FODO quad node names:")
    pprint(fodo_quad_names)

# Identify matching quads.
match_index_start = index_start
match_index_stop = lattice.getNodeIndex(lattice.getNodeForName("MEBT:QH10"))
matching_quad_names = []
for node in lattice.getNodes()[match_index_start : match_index_stop + 1]:
    if isinstance(node, Quad):
        matching_quad_names.append(node.getName())

# Set up optics controller.
optics_controller = pyorbit_sim.linac.OpticsController(lattice, matching_quad_names)

# Get quad gradient limits.
bounds = [linac.get_quad_kappa_limits(name) for name in matching_quad_names]
bounds = np.array(bounds)
bounds = bounds.T
bounds = optimize.Bounds(bounds[0], bounds[1])

# Set up matcher.
matcher = Matcher(
    lattice=lattice,
    bunch=bunch,
    optics_controller=optics_controller,
    fodo_quad_names=fodo_quad_names,
    index_start=index_start,
    index_stop=index_stop,
    save_freq=(100 if save else 0),
    verbose=False,
    outdir=(man.outdir if save else None),
)

# Compute matched optics.
matcher.match(
    bounds=bounds,
    method="trust-constr",
    options=dict(verbose=2),
)