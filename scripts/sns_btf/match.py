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
parser.add_argument("--quads", type=str, default="magnets/230904221417-quad_strengths_matched.dat")
parser.add_argument("--mstate", type=str, default=None)
    
# Lattice
parser.add_argument("--apertures", type=int, default=1)
parser.add_argument("--linac_tracker", type=int, default=1)
parser.add_argument("--max_drift", type=float, default=0.010)
parser.add_argument("--overlap", type=int, default=1)
parser.add_argument("--rf_freq", type=float, default=402.5e+06)

# Space charge (uniform density ellipsoid).
parser.add_argument("--spacecharge", type=str, choices=["fft", "ellipsoid"], default="ellipsoid")
parser.add_argument("--n_ellipsoids", type=int, default=5)

# Bunch
parser.add_argument("--bunch", type=str, default=None)
parser.add_argument("--charge", type=float, default=-1.0)  # [elementary charge units]
parser.add_argument("--current", type=float, default=0.042)  # [A]
parser.add_argument("--energy", type=float, default=0.0025)  # [GeV]
parser.add_argument("--mass", type=float, default=0.939294)  # [GeV / c^2]
parser.add_argument("--dist", type=str, default="wb", choices=["kv", "wb", "gs"])
parser.add_argument("--n_parts", type=int, default=20000)
parser.add_argument("--decorr", type=int, default=0)
parser.add_argument("--rms_equiv", type=int, default=0)

# Optimizer
parser.add_argument("--start", type=str, default=None) 
parser.add_argument("--start_pos", type=float, default=None)
parser.add_argument("--stride", type=float, default=0.010)
parser.add_argument("--maxiter", type=int, default=500)
parser.add_argument("--alpha", type=float, default=0.01)
parser.add_argument("--verbose", type=int, default=0)
parser.add_argument("--callback_freq", type=int, default=10)

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
# Set optics from file.
if args.mstate:
    filename = os.path.join(input_dir, args.mstate)
    linac.update_quads_from_mstate(filename, value_type="current")
if args.quads:
    filename = os.path.join(input_dir, args.quads)
    linac.set_quads_from_file(filename, comment="#", verbose=args.verbose)
if args.overlap:
    linac.set_overlapping_pmq_fields(z_step=args.max_drift, verbose=True)
if args.spacecharge == "ellipsoid":
    linac.add_uniform_ellipsoid_space_charge_nodes(
        n_ellipsoids=args.n_ellipsoids,
        path_length_min=args.max_drift,
    )
else:
    linac.add_space_charge_nodes(
        grid_size_x=64,
        grid_size_y=64,
        grid_size_z=64,
        n_bunches=3,
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
    n=args.n_parts,
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

bunch0 = Bunch()
bunch.copyBunchTo(bunch0)


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
fodo_quads = [lattice.getNodeForName(name) for name in linac.quad_names_fodo]
if _mpi_rank == 0:
    for node in fodo_quads:
        print("fodo quad: {}".format(node.getName()))

# Identify quads to vary.
match_index_start = index_start
match_index_stop = lattice.getNodeIndex(lattice.getNodeForName("MEBT:QH10"))
var_quads = []
for node in lattice.getNodes()[match_index_start : match_index_stop + 1]:
    if isinstance(node, Quad) and (node not in fodo_quads):
        var_quads.append(node)
if _mpi_rank == 0:
    for node in var_quads:
        print("opt quad: {}".format(node.getName()))
        
# Get bounds on matching quad strengths.
bounds = [linac.get_quad_kappa_limits(node.getName()) for node in var_quads]
bounds = np.array(bounds).T
bounds = optimize.Bounds(bounds[0], bounds[1])


# FODO matching
# ------------------------------------------------------------------------------

class OpticsController:
    def __init__(self, lattice=None, quad_nodes=None):
        self.lattice = lattice
        self.quad_nodes = quad_nodes
        self.quad_names = [node.getName for node in quad_nodes]

    def get_quad_strengths(self):
        return np.array([node.getParam("dB/dr") for node in self.quad_nodes])

    def set_quad_strengths(self, x):
        for i, node in enumerate(self.quad_nodes):
            node.setParam("dB/dr", x[i])
    
    
class BeamSizeMonitor:
    def __init__(self, fodo_nodes=None, stride=0.1, verbose=False):
        self.fodo_nodes = fodo_nodes
        self.stride = stride
        self.verbose = verbose
        self.reset()
        
    def reset(self):
        self.fodo_sizes_x = []
        self.fodo_sizes_y = []
        self.max_size_x = 0.0
        self.max_size_y = 0.0
        self.max_size_x_node = ""
        self.max_size_y_node = ""
        self.position = 0.0
        self.monitor_max_beam_size = True

    def action(self, params_dict):    
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
            
        bunch = params_dict["bunch"]
        node = params_dict["node"]
        position = params_dict["path_length"]
        
        if self.fodo_nodes and node == self.fodo_nodes[0]:
            self.monitor_max_beam_size = False
            
        # Pass if one of the following is true:
        # - we are not at a FODO node and not enough distance since last update
        # - we are not at a FODO node and we are inside FODO channel
        is_fodo = self.fodo_nodes and (node in self.fodo_nodes)
        if not is_fodo:
            if not self.monitor_max_beam_size:
                return
            if position - self.position < self.stride:
                return
        self.position = position
                    
        twiss_analysis = BunchTwissAnalysis()
        twiss_analysis.computeBunchMoments(bunch, 2, 0, 0)
        size_x = 1000.0 * np.sqrt(twiss_analysis.getCorrelation(0, 0))
        size_y = 1000.0 * np.sqrt(twiss_analysis.getCorrelation(2, 2))
        if is_fodo:
            self.fodo_sizes_x.append(size_x)
            self.fodo_sizes_y.append(size_y)
        else:
            if size_x > self.max_size_x:
                self.max_size_x = size_x
                self.max_size_x_node = node.getName()
            if size_y > self.max_size_y:
                self.max_size_y = size_y
                self.max_size_y_node = node.getName()
        if _mpi_rank == 0 and self.verbose:
            if is_fodo:
                print("xrms={:<7.3f} yrms={:<7.3f} node={}".format(size_x, size_x, node.getName()))

                
class Matcher:
    def __init__(
        self, 
        lattice=None, 
        bunch=None,
        index_start=0, 
        index_stop=-1, 
        verbose=0,
        fodo_quads=None,
        var_quads=None,
        stride=0.010,
        alpha=0.01,
        callback_freq=10,
        vis_freq=10,
    ):
        self.lattice = lattice
        self.bunch = bunch
        self.index_start = index_start
        self.index_stop = index_stop
        self.verbose = verbose
        self.alpha = alpha
        self.fodo_quads = fodo_quads
        self.var_quads = var_quads
        self.stride = stride
        self.iteration = 0
        self.callback_freq = callback_freq
        self.monitor = BeamSizeMonitor(
            fodo_nodes=self.fodo_quads, 
            stride=self.stride,
            verbose=self.verbose
        )
        self.optics_controller = OpticsController(self.lattice, self.var_quads)
        self.vis_freq = vis_freq
        
    def track(self, verbose=0):
        bunch_in = Bunch()
        self.bunch.copyBunchTo(bunch_in)
        self.monitor.reset()
        self.monitor.verbose = verbose > 1
        action_container = AccActionsContainer()
        action_container.addAction(self.monitor.action, AccActionsContainer.ENTRANCE)
        self.lattice.trackBunch(
            bunch_in,
            actionContainer=action_container,
            index_start=self.index_start,
            index_stop=self.index_stop,
        )
        
    def objective(self, x, verbose=0):
        self.optics_controller.set_quad_strengths(x)
        self.track(verbose=verbose)
        cost = 0.0
        for i in range(2):
            cost += 0.5 * np.var(self.monitor.fodo_sizes_x[i::2])
            cost += 0.5 * np.var(self.monitor.fodo_sizes_y[i::2])
        cost += 0.5 * args.alpha * self.monitor.max_size_x
        cost += 0.5 * args.alpha * self.monitor.max_size_y
        return cost
    
    def callback(self, x, state):
        if self.iteration % self.callback_freq == 0:
            self.track(verbose=2)
            if _mpi_rank == 0:
                for i in range(len(x)):
                    print("{}={:0.3f}".format(self.var_quads[i].getName(), x[i]))
                if self.fodo_quads:
                    print("var_x (even) = {}".format(np.var(self.monitor.fodo_sizes_x[0::2])))
                    print("var_y (even) = {}".format(np.var(self.monitor.fodo_sizes_y[0::2])))
                    print("var_x (odd) = {}".format(np.var(self.monitor.fodo_sizes_x[1::2])))
                    print("var_y (odd) = {}".format(np.var(self.monitor.fodo_sizes_y[1::2])))
                print("max_size_x = {:0.3f} (node={})".format(self.monitor.max_size_x, self.monitor.max_size_x_node))
                print("max_size_x = {:0.3f} (node={})".format(self.monitor.max_size_y, self.monitor.max_size_y_node))
            self.save()
            self.plot()
        self.iteration += 1
        
    def save(self):
        filename = "quad_strengths_{:05.0f}.dat".format(self.iteration)
        filename = man.get_filename(filename)
        linac.save_quad_strengths(filename)
        
    def plot(self):
        
        class _Monitor:
            def __init__(self):
                self.sizes = []
                self.positions = []
            def action(self, params_dict):
                twiss_analysis = BunchTwissAnalysis()
                twiss_analysis.computeBunchMoments(params_dict["bunch"], 2, 0, 0)
                size_x = 1000.0 * np.sqrt(twiss_analysis.getCorrelation(0, 0))
                size_y = 1000.0 * np.sqrt(twiss_analysis.getCorrelation(2, 2))
                self.sizes.append([size_x, size_y])
                self.positions.append(params_dict["path_length"])
                
        _monitor = _Monitor()
        bunch_in = Bunch()
        bunch0.copyBunchTo(bunch_in)
        action_container = AccActionsContainer()
        action_container.addAction(_monitor.action, AccActionsContainer.ENTRANCE)
        self.lattice.trackBunch(
            bunch_in,
            actionContainer=action_container,
        )   
        positions = np.array(_monitor.positions)
        sizes = np.array(_monitor.sizes)
        fig, ax = plt.subplots(figsize=(10.0, 2.25))
        ax.plot(positions, sizes[:, 0])
        ax.plot(positions, sizes[:, 1])
        for node in (self.lattice.getQuads() + self.fodo_quads):
            start, stop = self.lattice.getNodePositionsDict()[node]
            ax.axvspan(start, stop, color="black", alpha=0.05, ec="None")
        ax.set_xlabel("position [m]")
        ax.set_ylabel("beam size [mm]")
        filename = "fig_sizes_{:05.0f}.png".format(self.iteration)
        filename = man.get_filename(filename)
        plt.savefig(filename)
        plt.close("all")
    
    def match(self, bounds=None, maxiter=500):
        result = optimize.minimize(
            self.objective,
            self.optics_controller.get_quad_strengths(), 
            method="trust-constr", 
            bounds=bounds, 
            options=dict(
                maxiter=maxiter,
                verbose=(2 if _mpi_rank == 0 else 0)
            ),
            callback=self.callback,
        )    
        return result
    

# Track to start node (use separate lattice with PIC tracking).
_linac = SNS_BTF(
    coef_filename=os.path.join(input_dir, args.coef), 
    rf_frequency=args.rf_freq,
)
_linac.init_lattice(
    xml_filename=os.path.join(input_dir, args.xml),
    sequences=[
        "MEBT1",
        "MEBT2",
    ],
    max_drift_length=args.max_drift,
)
if args.mstate:
    filename = os.path.join(input_dir, args.mstate)
    _linac.update_quads_from_mstate(filename, value_type="current")
if args.quads:
    filename = os.path.join(input_dir, args.quads)
    _linac.set_quads_from_file(filename, comment="#", verbose=args.verbose)
if args.overlap:
    _linac.set_overlapping_pmq_fields(z_step=args.max_drift, verbose=True)
if args.spacecharge:
    _linac.add_space_charge_nodes(
        grid_size_x=64,
        grid_size_y=64,
        grid_size_z=64,
        n_bunches=3,
        path_length_min=args.max_drift,
    )
if args.apertures:
    _linac.add_aperture_nodes(drift_step=0.1, verbose=True)
if args.linac_tracker:
    _linac.set_linac_tracker(args.linac_tracker)
_lattice = linac.lattice
_lattice.trackBunch(bunch, index_start=0, index_stop=(index_start - 1))    
    
    
matcher = Matcher(
    lattice=lattice, 
    bunch=bunch,
    index_start=index_start, 
    index_stop=index_stop, 
    fodo_quads=fodo_quads,
    var_quads=var_quads,
    stride=args.stride,
    alpha=args.alpha,
    callback_freq=args.callback_freq,
)

result = matcher.match(bounds=bounds, maxiter=args.maxiter)
if _mpi_rank == 0:
    print(result)
    
if args.save:
    linac.save_quad_strengths(man.get_filename("quad_strengths.dat"))
    

    
# Optimize final focusing quads.
# ------------------------------------------------------------------------------


class BeamSizeMonitor:
    def __init__(self, stride=0.010, verbose=False):
        self.stride = stride
        self.verbose = verbose
        self.reset()
        
    def reset(self):
        self.max_size_x = 0.0
        self.max_size_y = 0.0
        self.max_size_x_node = ""
        self.max_size_y_node = ""
        self.position = 0.0

    def action(self, params_dict):    
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
            
        bunch = params_dict["bunch"]
        node = params_dict["node"]
        position = params_dict["path_length"]
        
        if position - self.position < self.stride:
            return
        self.position = position
                    
        twiss_analysis = BunchTwissAnalysis()
        twiss_analysis.computeBunchMoments(bunch, 2, 0, 0)
        size_x = 1000.0 * np.sqrt(twiss_analysis.getCorrelation(0, 0))
        size_y = 1000.0 * np.sqrt(twiss_analysis.getCorrelation(2, 2))
        if size_x > self.max_size_x:
            self.max_size_x = size_x
            self.max_size_x_node = node.getName()
        if size_y > self.max_size_y:
            self.max_size_y = size_y
            self.max_size_y_node = node.getName()
        if _mpi_rank == 0 and self.verbose:
            print("xrms={:<7.3f} yrms={:<7.3f} node={}".format(size_x, size_x, node.getName()))

class Matcher:
    def __init__(
        self, 
        lattice=None, 
        bunch=None,
        index_start=0, 
        index_stop=-1, 
        verbose=0,
        var_quads=None,
        stride=0.010,
        alpha=0.01,
        callback_freq=10,
    ):
        self.lattice = lattice
        self.bunch = bunch
        self.index_start = index_start
        self.index_stop = index_stop
        self.verbose = verbose
        self.alpha = alpha
        self.var_quads = var_quads
        self.stride = stride
        self.iteration = 0
        self.callback_freq = callback_freq
        self.monitor = BeamSizeMonitor(
            stride=self.stride,
            verbose=self.verbose
        )
        self.optics_controller = OpticsController(self.lattice, self.var_quads)
        
    def track(self, verbose=0):
        bunch_in = Bunch()
        self.bunch.copyBunchTo(bunch_in)
        self.monitor.reset()
        self.monitor.verbose = verbose > 1
        action_container = AccActionsContainer()
        action_container.addAction(self.monitor.action, AccActionsContainer.ENTRANCE)
        self.lattice.trackBunch(
            bunch_in,
            actionContainer=action_container,
            index_start=self.index_start,
            index_stop=self.index_stop,
        )
        
    def objective(self, x, verbose=0):
        self.optics_controller.set_quad_strengths(x)
        self.track(verbose=verbose)
        cost = 0.0
        cost += 0.5 * self.monitor.max_size_x
        cost += 0.5 * self.monitor.max_size_y
        return cost
    
    def match(self, bounds=None, maxiter=500):
        result = optimize.minimize(
            self.objective,
            self.optics_controller.get_quad_strengths(), 
            method="trust-constr", 
            bounds=bounds, 
            options=dict(
                maxiter=maxiter,
                verbose=(2 if _mpi_rank == 0 else 0)
            ),
        )    
        return result



if _mpi_rank == 0:
    print("Tracking to end of FODO line.")
    
# Track the bunch to the end of the FODO line.
index_stop = lattice.getNodeIndex(fodo_quads[-1])
lattice.trackBunch(bunch, index_start=0, index_stop=index_stop)
index_start = index_stop + 1
index_stop = lattice.getNodeIndex(lattice.getNodeForName("MEBT:HZ33a")) + 3

# Identify quads to vary.
var_quads = []
for node in lattice.getNodes()[index_start : index_stop + 1]:
    if isinstance(node, Quad) and (node not in fodo_quads):
        var_quads.append(node)
if _mpi_rank == 0:
    for node in var_quads:
        print("opt quad: {}".format(node.getName()))
        
# Get bounds on matching quad strengths.
bounds = [linac.get_quad_kappa_limits(node.getName()) for node in var_quads]
bounds = np.array(bounds).T
bounds = optimize.Bounds(bounds[0], bounds[1])
        
matcher = Matcher(
    lattice=lattice, 
    bunch=bunch,
    index_start=index_start, 
    index_stop=index_stop, 
    var_quads=var_quads,
    stride=0.001,
    alpha=1.0,
    callback_freq=args.callback_freq,
)

x = matcher.optics_controller.get_quad_strengths()

result = matcher.match(bounds=bounds, maxiter=args.maxiter)

if _mpi_rank == 0:
    print(result)
    
if args.save:
    linac.save_quad_strengths(man.get_filename("quad_strengths.dat"))