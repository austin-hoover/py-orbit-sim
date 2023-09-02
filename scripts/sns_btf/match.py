"""Match an input beam to the FODO line.

This script varies all the quads before the FODO line (after a specified start position) 
to generate a periodic beam envelope in the FODO line. It also penalizes large beam sizes
before the FODO line. Finally, it optimizes the remaining quads after the FODO line.

We use particle tracking, so all nonlinear effects can be left on. Around 10,000 
particles is enough for accurate rms sizes. Tracking is done using the uniform density 
ellipsoid space charge solver. This solver is very fast and gives approximately correct
rms beam sizes throughout the lattice. 

The optimization calls scipy.optimize.minimize and works with MPI.

TO DO:
    * Use argparse.
    
"""
from __future__ import print_function
import argparse
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
import yaml

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

from sns_btf import SNS_BTF

import pyorbit_sim
from pyorbit_sim.utils import ScriptManager


# Parse command line arguments
# --------------------------------------------------------------------------------------
parser = argparse.ArgumentParser("sim")

parser.add_argument("--nparts", type=int, default=10000)
parser.add_argument("--alpha", type=float, default=0.01)
parser.add_argument("--verbose", type=int, default=0)

args = parser.parse_args()


    
# Setup
# --------------------------------------------------------------------------------------
file_dir = os.path.dirname(os.path.realpath(__file__))

# Load config file.
file = open(os.path.join(file_dir, "config.yaml"), "r")
config = yaml.safe_load(file)
file.close()

# Set input/output directories.
input_dir = os.path.join(file_dir, "data_input")
output_dir = os.path.join(file_dir, config["output_dir"])

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

# Create output directory and save script info.
man = ScriptManager(
    datadir=output_dir,
    path=pathlib.Path(__file__),
    timestamp=timestamp,
    datestamp=datestamp,
)
if _mpi_rank == 0:
    man.make_outdir()
    log = man.get_logger()
    for key, val in man.get_info().items():
        log.info("{} {}".format(key, val))
    man.save_script_copy()



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
    "/home/46h/projects/btf/sim/sns_rfq/parmteq/2021-01-01_benchmark/data/",
    "bunch_RFQ_output_1.00e+05.dat",
)
mass = 0.939294  # [GeV / c^2]
charge = -1.0  # [elementary charge units]
kin_energy = 0.0025  # [GeV]
current = 0.042  # [A]
n_parts = args.nparts  # max number of particles
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


# Match to FODO channel
# ------------------------------------------------------------------------------

# Update start/stop node indices.
index_start = 0
index_stop = lattice.getNodeIndex(lattice.getNodeForName("MEBT:QH30"))
position_offset, _ = lattice.getNodePositionsDict()[lattice.getNodes()[index_start]]
stride = 0.001

# Identify matching quads.
match_index_start = index_start
match_index_stop = lattice.getNodeIndex(lattice.getNodeForName("MEBT:QH10"))
matching_quad_names = []
for node in lattice.getNodes()[match_index_start : match_index_stop + 1]:
    if isinstance(node, Quad):
        matching_quad_names.append(node.getName())
        
# Identify fodo quads.
fodo_quad_names = linac.quad_names_fodo
fodo_quad_nodes = [lattice.getNodeForName(name) for name in linac.quad_names_fodo]

# Create optics controller.
optics_controller = pyorbit_sim.linac.OpticsController(lattice, matching_quad_names)
bounds = [linac.get_quad_kappa_limits(name) for name in matching_quad_names]
bounds = np.array(bounds)
bounds = bounds.T
bounds = optimize.Bounds(bounds[0], bounds[1])

# Also minimize average beam size at these nodes:
other_nodes = []
position_start, _ = lattice.getNodePositionsDict()[lattice.getNodes()[index_start]]
position_stop, _ = lattice.getNodePositionsDict()[fodo_quad_nodes[0]]
for position in np.linspace(position_start, position_stop, 20):
    node = lattice.getNodeForPosition(position)[0]
    other_nodes.append(node)
other_node_names = [node.getName() for node in other_nodes]

for node in other_nodes:
    if _mpi_rank == 0:
        print(node.getName())


# Run optimizer.

class BeamSizeMonitor:
    def __init__(self, stride=0.5, verbose=False, fodo_nodes=None, other_nodes=None):
        self.verbose = verbose
        self.nodes_other = other_nodes
        self.nodes_fodo = fodo_nodes
        self.reset()
        
    def reset(self):
        self.sizes_fodo = []
        self.sizes_other = []

    def action(self, params_dict):    
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
            
        bunch = params_dict["bunch"]
        node = params_dict["node"]
        position = params_dict["path_length"]
        
        is_fodo = self.nodes_fodo and node in self.nodes_fodo
        is_other = self.nodes_other and node in self.nodes_other
        
        if not (is_fodo or is_other):
            return
            
        twiss_analysis = BunchTwissAnalysis()
        twiss_analysis.computeBunchMoments(bunch, 2, 0, 0)
        size_x = 1000.0 * np.sqrt(twiss_analysis.getCorrelation(0, 0))
        size_y = 1000.0 * np.sqrt(twiss_analysis.getCorrelation(2, 2))
        if is_fodo:
            self.sizes_fodo.append([size_x, size_x])   
        else:
            self.sizes_other.append([size_x, size_y])        
        if _mpi_rank == 0 and self.verbose:
            print("xrms={:<7.3f} yrms={:<7.3f} node={}".format(size_x, size_x, node.getName()))


def track(lattice=None, bunch=None, monitor=None, index_start=0, index_stop=-1):
    bunch_in = Bunch()
    bunch.copyBunchTo(bunch_in)
    monitor.reset()
    action_container = AccActionsContainer()
    action_container.addAction(monitor.action, AccActionsContainer.ENTRANCE)
    lattice.trackBunch(
        bunch_in,
        actionContainer=action_container,
        index_start=index_start,
        index_stop=index_stop,
    )
    sizes_fodo = np.array(monitor.sizes_fodo)
    sizes_other = np.array(monitor.sizes_other)
    return sizes_fodo, sizes_other


def objective(x, monitor, alpha=0.01, verbose=False):
    optics_controller.set_quad_strengths(x)
    sizes_fodo, sizes_other = track(
        lattice, 
        bunch, 
        monitor,
        index_start=index_start, 
        index_stop=index_stop, 
    )
    cost = 0.0
    for i in range(2):
        cost += np.var(sizes_fodo[i::2, 0])
        cost += np.var(sizes_fodo[i::2, 1])
    cost += alpha * np.mean(sizes_other[:, 0])
    cost += alpha * np.mean(sizes_other[:, 1])
    if verbose and _mpi_rank == 0:
        print("cost={:0.2e}".format(cost))
    return cost


if _mpi_rank == 0:
    print("Matching to FODO channel")
    print("FODO quad node names:")
    pprint(fodo_quad_names)
    print("Quads being optimized:")
    pprint(matching_quad_names)
    
monitor = BeamSizeMonitor(fodo_nodes=fodo_quad_nodes, other_nodes=other_nodes, verbose=args.verbose)
result = optimize.minimize(
    objective,
    optics_controller.get_quad_strengths(), 
    method="trust-constr", 
    bounds=bounds, 
    args=(monitor, args.alpha, args.verbose),
    options=dict(
        maxiter=100,
        verbose=(2 if _mpi_rank == 0 else 0)
    )
)
linac.save_quad_strengths(man.get_filename("quad_strengths.dat"))



# Optimize final focusing quads.
# ------------------------------------------------------------------------------
   
# Track the bunch to the end of the FODO line.
if _mpi_rank == 0:
    print("Tracking to end of FODO line.")
    print("Optimizing final focusing quads.")
index_stop = lattice.getNodeIndex(lattice.getNodeForName("MEBT:QH30")) - 1
lattice.trackBunch(bunch, index_start=0, index_stop=index_stop)
index_start = index_stop
index_stop = -1

# Optimize the remaining quads.
quad_names = []
for node in lattice.getNodes()[index_start:]:
    if isinstance(node, Quad):
        quad_names.append(node.getName())
optics_controller = pyorbit_sim.linac.OpticsController(lattice, quad_names)
bounds = [linac.get_quad_kappa_limits(name) for name in quad_names]
bounds = np.array(bounds).T
bounds = optimize.Bounds(bounds[0], bounds[1])


def objective(x, monitor, verbose=False):
    optics_controller.set_quad_strengths(x)
    _, sizes = track(
        lattice, 
        bunch, 
        monitor,
        index_start=index_start, 
        index_stop=index_stop, 
    )
    cost = 0.0
    cost += alpha * np.mean(sizes[:, 0])
    cost += alpha * np.mean(sizes[:, 1])
    if verbose and _mpi_rank == 0:
        print("cost={:0.2e}".format(cost))
    return cost


if _mpi_rank == 0:
    print("Matching to FODO channel")
    print("FODO quad node names:")
    pprint(linac.quad_names_fodo)
    
nodes = lattice.getNodes()[index_start : index_stop : 20]
monitor = BeamSizeMonitor(other_nodes=nodes, verbose=args.verbose)
result = optimize.minimize(
    objective,
    optics_controller.get_quad_strengths(), 
    method="trust-constr", 
    bounds=bounds, 
    args=(monitor, args.verbose),
    options=dict(
        maxiter=500,
        verbose=(2 if _mpi_rank == 0 else 0)
    )
)
linac.save_quad_strengths(man.get_filename("quad_strengths.dat"))