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

from SNS_BTF import BeamSizeMonitor
from SNS_BTF import Matcher
from SNS_BTF import OpticsController
from SNS_BTF import SNS_BTF

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
output_dir = os.path.join(file_dir, "data_output")
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
    script_path_in_outdir=False,
)
    
    
# Lattice
# ------------------------------------------------------------------------------

# Settings
xml_filename = os.path.join(input_dir, "xml/btf_lattice_straight.xml")
coef_filename = os.path.join(input_dir, "magnets/default_i2gl_coeff_straight.csv")
sequences = [
    "MEBT1",
    "MEBT2",
]
max_drift_length = 0.010  # [m]

# Create lattice for tracking to the matching section.
linac = SNS_BTF(coef_filename=coef_filename, rf_frequency=402.5e+06)
linac.init_lattice(
    xml_filename=xml_filename,
    sequences=sequences,
    max_drift_length=max_drift_length,
)
linac.add_space_charge_nodes(
    grid_size_x=64,
    grid_size_y=64,
    grid_size_z=64,
    path_length_min=max_drift_length,
    n_bunches=3,
)
linac.add_aperture_nodes(drift_step=0.1, verbose=True)
linac.set_linac_tracker(False)
lattice = linac.lattice
node_positions_dict = lattice.getNodePositionsDict()

# for (node, (start, stop)) in sorted(node_positions_dict.items(), key=lambda item: item[1][0]):
#     print(node.getName(), start, stop)


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
n_parts = int(2.00e+04)  # max number of particles
intensity = pyorbit_sim.bunch_utils.get_intensity(current, linac.rf_frequency)

# Initialize the bunch.
bunch = Bunch()
bunch.mass(mass)
bunch.charge(charge)
bunch.getSyncParticle().kinEnergy(kin_energy)
pyorbit_sim.bunch_utils.load(
    filename=filename,
    bunch=bunch,
    verbose=True,
)
bunch = pyorbit_sim.bunch_utils.set_centroid(bunch, centroid=0.0)
if n_parts is not None:
    bunch = pyorbit_sim.bunch_utils.downsample(
        bunch, 
        n=int(n_parts / _mpi_size),
        method="first", 
        verbose=True,
    )
bunch_size_global = bunch.getSizeGlobal()
macro_size = intensity / bunch_size_global
bunch.macroSize(macro_size)

if _mpi_rank == 0:
    print("bunch.getSizeGlobal() = {}".format(bunch_size_global))


# Tracking to matching section
# ------------------------------------------------------------------------------
            
stop_node_name = "MEBT:VT06"
index_start = 0
index_stop = lattice.getNodeIndex(lattice.getNodeForName(stop_node_name))
if _mpi_rank == 0:
    print(
        "Tracking from {} to {}".format(
            lattice.getNodes()[index_start].getName(),
            lattice.getNodes()[index_stop].getName(),
        )
    )

monitor = BeamSizeMonitor(verbose=True, stride=0.100)
action_container = AccActionsContainer()
action_container.addAction(monitor.action, AccActionsContainer.ENTRANCE)
action_container.addAction(monitor.action, AccActionsContainer.EXIT)
lattice.trackBunch(
    bunch,
    actionContainer=action_container,
    index_start=0,
    index_stop=index_stop,
)


# Matching to FODO channel
# ------------------------------------------------------------------------------

if _mpi_rank == 0:
    print("Matching to FODO channel")
    
# Create a new lattice with uniform ellipsoid space charge nodes (we cannot
# remove the original space charge nodes, and we cannot add new child nodes
# to the lattice.)
linac = SNS_BTF(coef_filename=coef_filename, rf_frequency=402.5e+06)
linac.init_lattice(
    xml_filename=xml_filename,
    sequences=sequences,
    max_drift_length=max_drift_length,
)
linac.add_uniform_ellipsoid_space_charge_nodes(
    n_ellipsoids=3,
    path_length_min=0.010,
)
linac.set_linac_tracker(False)
lattice = linac.lattice

# Update start/stop node indices.
index_start = lattice.getNodeIndex(lattice.getNodeForName(stop_node_name)) + 1
index_stop = lattice.getNodeIndex(lattice.getNodeForName("MEBT:QH30"))

# Identify FODO quads.
fodo_quad_names = ["MEBT:FQ{}".format(i) for i in range(11, 30)]

# Identify matching quads.
match_index_start = index_start
match_index_stop = lattice.getNodeIndex(lattice.getNodeForName("MEBT:QH10"))
matching_quad_names = []
for node in lattice.getNodes()[match_index_start : match_index_stop + 1]:
    if isinstance(node, Quad):
        matching_quad_names.append(node.getName())
    
# Set up optics controller.
optics_controller = pyorbit_sim.linac.OpticsController(lattice, matching_quad_names)
(lb, ub) = optics_controller.estimate_quad_bounds(scale=1.5)
bounds = optimize.Bounds(lb, ub)

# Set up matcher.
matcher = Matcher(
    lattice=lattice,
    bunch=bunch,
    optics_controller=optics_controller,
    fodo_quad_names=fodo_quad_names,
    index_start=index_start,
    index_stop=index_stop,
    save_freq=(25 if save else 0),
    verbose=False,
    prefix=man.prefix,
    outdir=man.outdir,
)

# Compute matched optics.
matcher.match(
    bounds=bounds,
    method="trust-constr",
    options=dict(verbose=2),
)