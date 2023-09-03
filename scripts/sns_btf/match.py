"""Match an input beam to the FODO line.

This script varies all the quads before the FODO line (after a specified start position) 
to generate a periodic beam envelope in the FODO line. It also penalizes large beam sizes
before the FODO line. Finally, it optimizes the remaining quads after the FODO line.

We use particle tracking, so all nonlinear effects can be left on. Around 10,000 
particles is enough for accurate rms sizes. Tracking is done using the uniform density 
ellipsoid space charge solver. This solver is very fast and gives approximately correct
rms beam sizes throughout the lattice. 

The optimization calls scipy.optimize.minimize and works with MPI.
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
from orbit.bunch_generators import GaussDist3D
from orbit.bunch_generators import KVDist3D
from orbit.bunch_generators import WaterBagDist3D
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


# Parse arguments
# --------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

# Input/output paths
parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--xml", type=str, default="xml/btf_lattice_straight.xml")
parser.add_argument("--coef", type=str, default="magnets/default_i2gl_coeff_straight.csv")
parser.add_argument("--mstate", type=str, default=None)
    
# Lattice
parser.add_argument("--apertures", type=int, default=1)
parser.add_argument("--linac_tracker", type=int, default=1)
parser.add_argument("--max_drift", type=float, default=0.010)
parser.add_argument("--overlap", type=int, default=1)
parser.add_argument("--rf_freq", type=float, default=402.5e+06)

# Space charge (uniform density ellipsoid).
parser.add_argument("--spacecharge", type=int, default=1)
parser.add_argument("--n_ellipsoids", type=int, default=3)

# Bunch
parser.add_argument("--bunch", type=str, default=None)
parser.add_argument("--charge", type=float, default=-1.0)  # [elementary charge units]
parser.add_argument("--current", type=float, default=0.042)  # [A]
parser.add_argument("--energy", type=float, default=0.0025)  # [GeV]
parser.add_argument("--mass", type=float, default=0.939294)  # [GeV / c^2]
parser.add_argument("--dist", type=str, default="wb", choices=["kv", "wb", "gs"])
parser.add_argument("--nparts", type=int, default=10000)
parser.add_argument("--decorr", type=int, default=0)
parser.add_argument("--rms_equiv", type=int, default=0)

# Optimizer
parser.add_argument("--start", type=str, default=None) 
parser.add_argument("--start_pos", type=float, default=None)
parser.add_argument("--n_monitor", type=int, default=30)
parser.add_argument("--maxiter", type=int, default=100)
parser.add_argument("--alpha", type=float, default=0.001)
parser.add_argument("--verbose", type=int, default=0)

parser.add_argument("--save", type=str, default=1)

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
if args.outdir is not None:
    output_dir = os.path.join(file_dir, args.outdir)

# MPI
_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

# Create output directory and save script info.
man = ScriptManager(datadir=output_dir, path=pathlib.Path(__file__))
if args.save and _mpi_rank == 0:
    man.make_outdir()
    log = man.get_logger(save=args.save, disp=True)
    for key, val in man.get_info().items():
        log.info("{} {}".format(key, val))
    log.info(args)
    man.save_script_copy()
    
    
# Lattice
# ------------------------------------------------------------------------------
linac = SNS_BTF(
    coef_filename=os.path.join(input_dir, args.coef), 
    rf_frequency=args.rf_freq,
)
linac.init_lattice(
    xml_filename=os.path.join(input_dir, args.xml),
    sequences=[
        "MEBT1",
        "MEBT2",
    ],
    max_drift_length=args.max_drift,
)
if args.mstate:
    filename = os.path.join(input_dir, args.mstate)
    linac.update_quads_from_mstate(filename, value_type="current")
if args.overlap:
    linac.set_overlapping_pmq_fields(z_step=args.max_drift, verbose=True)
if args.spacecharge:
    linac.add_uniform_ellipsoid_space_charge_nodes(
        n_ellipsoids=args.n_ellipsoids,
        path_length_min=args.max_drift,
    )
if args.apertures:
    linac.add_aperture_nodes(drift_step=0.1, verbose=True)
if args.linac_tracker:
    linac.set_linac_tracker(args.linac_tracker)
if args.save:
    linac.save_node_positions(man.get_filename("lattice_nodes.txt"))
    linac.save_lattice_structure(man.get_filename("lattice_structure.txt"))
lattice = linac.lattice


# Bunch
# ------------------------------------------------------------------------------

def_bunch_filename = os.path.join(
    "/home/46h/projects/btf/sim/sns_rfq/parmteq/2021-01-01_benchmark/data/",
    "bunch_RFQ_output_1.00e+05.dat"
)

dists = {
    "kv": KVDist3D, 
    "wb": WaterBagDist3D, 
    "gs": GaussDist3D,
}
dist = dists[args.dist]

bunch = Bunch()
bunch.mass(args.mass)
bunch.charge(args.charge)
bunch.getSyncParticle().kinEnergy(args.energy)

if args.bunch == "design":
    bunch = pyorbit_sim.bunch_utils.generate_norm_twiss(
        dist=dist,
        n=args.n,
        bunch=bunch,
        verbose=True,
        **config["bunch"]["twiss"]
    )
else:
    filename = args.bunch
    if filename is None:
        filename = def_bunch_filename
    bunch = pyorbit_sim.bunch_utils.load(
        filename=filename,
        bunch=bunch,
        verbose=True,
    )

bunch = pyorbit_sim.bunch_utils.set_centroid(bunch, centroid=0.0, verbose=True)

if args.rms_equiv:
    bunch = pyorbit_sim.bunch_utils.generate_rms_equivalent_bunch(
        dist=dist,
        bunch=bunch,
        verbose=True,
    )

bunch = pyorbit_sim.bunch_utils.downsample(
    bunch,
    n=args.nparts,
    method="first",
    conserve_intensity=True,
    verbose=True,
)

if args.decorr:
    bunch = pyorbit_sim.bunch_utils.decorrelate_x_y_z(bunch, verbose=True)

intensity = pyorbit_sim.bunch_utils.get_intensity(args.current, linac.rf_frequency)
bunch_size_global = bunch.getSizeGlobal()
macro_size = intensity / bunch_size_global
bunch.macroSize(macro_size)


# Optimizer settings
# ------------------------------------------------------------------------------

# Determine start/stop indices for optimization tracking.
start = args.start
if start is None:
    start = lattice.getNodes()[0].getName()
if args.start_pos is not None:
    start, _, _ = lattice.getNodeForPosition(args.start_pos)
stop = "MEBT:QH30"  # just after FODO line
index_start = lattice.getNodeIndex(lattice.getNodeForName(start))
index_stop = lattice.getNodeIndex(lattice.getNodeForName(stop))

# Identify fodo quads. Periodicity will be enfored at these nodes.
fodo_quad_names = linac.quad_names_fodo
fodo_quad_nodes = [lattice.getNodeForName(name) for name in linac.quad_names_fodo]
if _mpi_rank == 0:
    for name in fodo_quad_names:
        print("fodo quad: {}".format(name))

# Identify quads to vary.
match_index_start = index_start
match_index_stop = lattice.getNodeIndex(lattice.getNodeForName("MEBT:QH10"))
matching_quad_names = []
for node in lattice.getNodes()[match_index_start : match_index_stop + 1]:
    if isinstance(node, Quad) and (node not in fodo_quad_nodes):
        matching_quad_names.append(node.getName())
if _mpi_rank == 0:
    for name in matching_quad_names:
        print("opt quad: {}".format(name))
        
# Also minimize the average beam size at the following nodes. (This acts to 
# constrain the trajectories and minimize beam loss outside the FODO line).
other_nodes = []
nodes = lattice.getNodes()
node_pos_dict = lattice.getNodePositionsDict()
position_start, _ = node_pos_dict[nodes[index_start]]
position_stop, _ = node_pos_dict[fodo_quad_nodes[0]]
position_fodo_start, _ = node_pos_dict[fodo_quad_nodes[0]]
position_fodo_stop, _ = node_pos_dict[fodo_quad_nodes[-1]]
for position in np.linspace(position_start, position_stop, args.n_monitor):
    if not (position_fodo_start <= position <= position_fodo_stop):
        other_nodes.append(lattice.getNodeForPosition(position)[0])
if _mpi_rank == 0:
    for node in other_nodes:
        print("monitor node: {}".format(node.getName()))

# Create optics controller.
optics_controller = pyorbit_sim.linac.OpticsController(lattice, matching_quad_names)
bounds = [linac.get_quad_kappa_limits(name) for name in matching_quad_names]
bounds = np.array(bounds).T
bounds = optimize.Bounds(bounds[0], bounds[1])


# FODO matching
# ------------------------------------------------------------------------------
class BeamSizeMonitor:
    def __init__(self, fodo_nodes=None, other_nodes=None, verbose=False):
        self.other_nodes = other_nodes
        self.fodo_nodes = fodo_nodes
        self.verbose = verbose
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
        
        is_fodo = self.fodo_nodes and node in self.fodo_nodes
        is_other = self.other_nodes and node in self.other_nodes
        
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


def track(lattice=None, bunch=None, monitor=None, index_start=0, index_stop=-1, verbose=0):
    bunch_in = Bunch()
    bunch.copyBunchTo(bunch_in)
    monitor.reset()
    monitor.verbose = verbose > 1
    action_container = AccActionsContainer()
    action_container.addAction(monitor.action, AccActionsContainer.ENTRANCE)
    lattice.trackBunch(
        bunch_in,
        actionContainer=action_container,
        index_start=index_start,
        index_stop=index_stop,
    )
    return monitor


def objective(x, monitor, alpha=0.01, verbose=0):
    optics_controller.set_quad_strengths(x)
    monitor = track(
        lattice, 
        bunch, 
        monitor,
        index_start=index_start, 
        index_stop=index_stop, 
        verbose=verbose,
    )
    sizes_fodo = np.array(monitor.sizes_fodo)
    sizes_other = np.array(monitor.sizes_other)
    cost_m = 0.0
    for i in range(2):
        cost_m += np.var(sizes_fodo[i::2, 0])
        cost_m += np.var(sizes_fodo[i::2, 1])
    cost_s = np.mean(sizes_other)
    cost = cost_m + cost_s
    if verbose and _mpi_rank == 0:
        print("cost={:0.3e} M={:0.3e} S={:0.3e}".format(cost, cost_m, cost_s))
    return cost


if _mpi_rank == 0:
    print("Matching to FODO channel")
    print("Quads being optimized:")
    pprint(matching_quad_names)
    print("FODO quads (enforce periodicity):")
    pprint(fodo_quad_names)
    print("Also minimize average size at these nodes:")
    pprint([node.getName() for node in other_nodes])
    
monitor = BeamSizeMonitor(
    fodo_nodes=fodo_quad_nodes, 
    other_nodes=other_nodes, 
    verbose=args.verbose,
)

result = optimize.minimize(
    objective,
    optics_controller.get_quad_strengths(), 
    method="trust-constr", 
    bounds=bounds, 
    args=(monitor, args.alpha, args.verbose),
    options=dict(
        maxiter=args.maxiter,
        verbose=(2 if _mpi_rank == 0 else 0)
    )
)
if _mpi_rank == 0:
    print(result)
if args.save:
    linac.save_quad_strengths(man.get_filename("quad_strengths.dat"))
    
track(
    lattice, 
    bunch, 
    monitor,
    index_start=index_start, 
    index_stop=index_stop, 
    verbose=2,
)


# Optimize final focusing quads.
# ------------------------------------------------------------------------------

if _mpi_rank == 0:
    print("Tracking to end of FODO line.")
    
# Track the bunch to the end of the FODO line.
index_stop = lattice.getNodeIndex(fodo_quad_nodes[-1])
lattice.trackBunch(bunch, index_start=0, index_stop=index_stop)
index_start = index_stop + 1

# Optimize the remaining quads.
matching_quad_names = []
matching_quad_nodes = []
for node in lattice.getNodes()[index_start:]:
    if isinstance(node, Quad):
        matching_quad_nodes.append(node)
        matching_quad_names.append(node.getName())
        
optics_controller = pyorbit_sim.linac.OpticsController(lattice, matching_quad_names)
bounds = [linac.get_quad_kappa_limits(name) for name in matching_quad_names]
bounds = np.array(bounds).T
bounds = optimize.Bounds(bounds[0], bounds[1])

# Minimize the average beam size at the quads.
index_stop = lattice.getNodeIndex(lattice.getNodeForName("MEBT:HZ33a"))
nodes = lattice.getNodes()[:index_stop]
stride = max(1, int((len(nodes) - index_start) / 15))
idx = np.arange(index_start, len(nodes), stride)
other_nodes = [nodes[i] for i in idx]


def objective(x, monitor, verbose=0):
    optics_controller.set_quad_strengths(x)
    monitor = track(
        lattice, 
        bunch, 
        monitor,
        index_start=index_start, 
        index_stop=-1, 
        verbose=verbose,
    )
    sizes = np.array(monitor.sizes_other)
    cost = 0.0
    cost += np.max(sizes[:, 0])
    cost += np.max(sizes[:, 1])
    if verbose and _mpi_rank == 0:
        print("cost={:0.2e}".format(cost))
    return cost


if _mpi_rank == 0:
    print("Quads being optimized:")
    for i, node in enumerate(matching_quad_nodes):
        print("{} lb={} ub={}".format(node.getName(), bounds.lb[i], bounds.ub[i]))
    print("Minimize max beam size at these nodes:")
    pprint([node.getName() for node in other_nodes])
    
    
monitor = BeamSizeMonitor(other_nodes=other_nodes, verbose=args.verbose)
x0 = optics_controller.get_quad_strengths()
result = optimize.minimize(
    objective,
    x0,
    method="trust-constr", 
    bounds=bounds, 
    args=(monitor, args.verbose),
    options=dict(
        maxiter=args.maxiter,
        verbose=(2 if _mpi_rank == 0 else 0)
    )
)
if _mpi_rank == 0:
    print(result)
if args.save:
    linac.save_quad_strengths(man.get_filename("quad_strengths.dat"))
    
    
track(lattice, bunch, monitor, verbose=2)