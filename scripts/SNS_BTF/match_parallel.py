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
import scipy.optimize


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

from SNS_BTF import SNS_BTF

sys.path.append("../..")
import pyorbit_sim


# Setup
# --------------------------------------------------------------------------------------

# Set up directories.
file_dir = os.path.dirname(os.path.realpath(__file__))
input_dir = os.path.join(file_dir, "data_input")
output_dir = os.path.join(file_dir, "data_output")

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
coef_filename = os.path.join(input_dir, "magnets/default_i2gl_coeff.csv")
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
n_parts = int(1.00e+04)  # max number of particles
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

class Monitor:
    def __init__(self, verbose=True, stride=0.100):
        self.history = []
        self.verbose = verbose
        self.stride = stride
        self.position = 0.0
                
    def action(self, params_dict):
        position = params_dict["path_length"]
        if position - self.position < self.stride:
            return
        self.position = position

        node = params_dict["node"]
        bunch = params_dict["bunch"]
        twiss_analysis = BunchTwissAnalysis()
        order = 2
        twiss_analysis.computeBunchMoments(bunch, order, 0, 0)
        sig_xx = twiss_analysis.getCorrelation(0, 0)
        sig_yy = twiss_analysis.getCorrelation(2, 2)
        x_rms = 1000.0 * np.sqrt(sig_xx)
        y_rms = 1000.0 * np.sqrt(sig_yy)
        n_parts = bunch.getSizeGlobal()
        if _mpi_rank == 0 and self.verbose:
            print(
                "s={:<7.3f} xrms={:<7.3f} yrms={:<7.3f} nparts={:07.0f} node={}".format(
                    position, x_rms, y_rms, n_parts, node.getName()
                )
            )
            
            
            
# stop_node_name = "MEBT:VS06"
stop_node_name = "MEBT:VT06"
index_start = 0
index_stop = lattice.getNodeIndex(lattice.getNodeForName(stop_node_name))
if _mpi_rank == 0:
    print("Tracking through {}".format(lattice.getNodes()[index_stop].getName()))

monitor = Monitor(verbose=True, stride=0.100)
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

class OpticsController:
    """Sets quadrupole strengths."""
    def __init__(self, lattice=None, quad_names=None):
        self.lattice = lattice
        self.quad_names = quad_names
        self.quad_nodes = [lattice.getNodeForName(name) for name in quad_names]

    def get_quad_strengths(self):
        return [node.getParam("dB/dr") for node in self.quad_nodes]

    def set_quad_strengths(self, x):
        for i, node in enumerate(self.quad_nodes):
            node.setParam("dB/dr", x[i])

    def get_quad_bounds(self, scale=0.5):
        """Return (lower_bounds, upper_bounds) for quad strengths.
        
        `scale` determines the max strength relative to current set point.
        """
        lb, ub = [], []
        for kq in self.get_quad_strengths():
            sign = np.sign(kq)
            lo = 0.0
            hi = scale * np.abs(kq)
            if sign < 0:
                lo = -hi
                hi = 0.0
            lb.append(lo)
            ub.append(hi)
        return (lb, ub)
    
class Monitor:
    """Tracks the rms beam envelope oscillations."""
    def __init__(self, node_names=None, position_offset=0.0, verbose=False):
        """
        node_names : List of node names to observe. If None, observe every step.
        """
        self.node_names = node_names
        self.history = []
        self.position_offset = position_offset
        self.verbose = verbose

    def action(self, params_dict):
        bunch = params_dict["bunch"]
        node = params_dict["node"]
        position = params_dict["path_length"] + self.position_offset
        if self.node_names and (node.getName() not in self.node_names):
            return
        twiss_analysis = BunchTwissAnalysis()
        order = 2
        twiss_analysis.computeBunchMoments(bunch, order, 0, 0)
        sig_xx = twiss_analysis.getCorrelation(0, 0)
        sig_yy = twiss_analysis.getCorrelation(2, 2)
        x_rms = 1000.0 * np.sqrt(sig_xx)
        y_rms = 1000.0 * np.sqrt(sig_yy)
        self.history.append([position, x_rms, y_rms])
        if _mpi_rank == 0:
            if self.verbose:
                print("xrms={:<7.3f} yrms={:<7.3f} node={}".format(x_rms, y_rms, node.getName()))
                
                
class Optimizer:
    """Matches the beam to the FODO channel."""
    def __init__(
        self,
        lattice=None,
        optics_controller=None,
        fodo_quad_names=None,
        index_start=0,
        index_stop=-1,
        save_freq=None,
        verbose=False,
    ):
        self.lattice = lattice
        self.optics_controller = optics_controller
        self.fodo_quad_names = fodo_quad_names
        self.index_start = index_start
        self.index_stop = index_stop
        self.position_offset, _ = lattice.getNodePositionsDict()[lattice.getNodes()[index_start]]
        self.save_freq = save_freq
        self.verbose = verbose
        self.count = 0
        
    def track(self, dense=False, verbose=False):
        """Return (x_rms, y_rms) at each FODO quad."""
        bunch_in = Bunch()
        bunch.copyBunchTo(bunch_in)
        
        monitor_node_names = None if dense else fodo_quad_names
        monitor = Monitor(
            node_names=monitor_node_names, 
            position_offset=self.position_offset, 
            verbose=verbose,
        )
        action_container = AccActionsContainer()
        action_container.addAction(monitor.action, AccActionsContainer.ENTRANCE)
        self.lattice.trackBunch(
            bunch_in,
            actionContainer=action_container,
            index_start=self.index_start,
            index_stop=self.index_stop,
        )
        history = np.array(monitor.history)
        return history
    
    def save_data(self):
        history = self.track(dense=True, verbose=False)
        positions, x_rms, y_rms = history.T
        
        if _mpi_rank == 0:
        
            fig, ax = plt.subplots(figsize=(7.0, 2.5), tight_layout=True)
            kws = dict()
            ax.plot(positions, x_rms, label="x", **kws)
            ax.plot(positions, y_rms, label="y", **kws)
            if self.count == 0:
                self.ymax = ax.get_ylim()[1]
            ax.set_ylim((0.0, self.ymax))

            node_start = self.lattice.getNodes()[self.index_start]
            for node in optics_controller.quad_nodes:
                start, stop = self.lattice.getNodePositionsDict()[node]
                ax.axvspan(start, stop, color="black", alpha=0.075, ec="None")

            ax.set_xlabel("Position [m]")
            ax.set_ylabel("RMS size [mm]")
            ax.legend(loc="upper right")
            filename = "envelope_{:06.0f}.png".format(self.count)
            filename = man.get_filename(filename)
            plt.savefig(filename, dpi=100)
            plt.close()

    def objective(self, x, stop):
        """Return variance of period-by-period beam sizes."""
        stop = orbit_mpi.MPI_Bcast(stop, mpi_datatype.MPI_INT, 0, _mpi_comm)    
        cost = 0.0
        if stop == 0:
            x = orbit_mpi.MPI_Bcast(x.tolist(), mpi_datatype.MPI_DOUBLE, 0, _mpi_comm) 

            self.optics_controller.set_quad_strengths(x)

            history = self.track(dense=False, verbose=self.verbose)
            positions, x_rms, y_rms = history.T

            cost_ = 0.0
            if _mpi_rank == 0:
                for i in range(2):
                    cost_ += np.var(x_rms[i::2])
                    cost_ += np.var(y_rms[i::2])

            cost = orbit_mpi.MPI_Bcast(cost_, mpi_datatype.MPI_DOUBLE, 0, _mpi_comm)    
                        
            if self.verbose and _mpi_rank == 0:
                print("cost={}".format(cost))
                
            if self.save_freq and (self.count % self.save_freq == 0):
                self.save_data()
            self.count += 1

        return cost


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
fodo_quad_names = ["MEBT:FQ{}".format(i) for i in range(11, 34)]

# Identify matching quads.
match_index_start = index_start
match_index_stop = lattice.getNodeIndex(lattice.getNodeForName("MEBT:QH10"))
matching_quad_names = []
for node in lattice.getNodes()[match_index_start : match_index_stop + 1]:
    if isinstance(node, Quad):
        matching_quad_names.append(node.getName())
if _mpi_rank == 0:
    print("Matching quads:")
    print(matching_quad_names)

# Set up optimizer.
optics_controller = OpticsController(lattice, matching_quad_names)
optimizer = Optimizer(
    lattice=lattice,
    optics_controller=optics_controller,
    fodo_quad_names=fodo_quad_names,
    index_start=index_start,
    index_stop=index_stop,
    save_freq=25,
    verbose=False,
)
objective = optimizer.objective

x0 = optics_controller.get_quad_strengths()
x0 = np.array(x0)
lb, ub = optics_controller.get_quad_bounds()
if _mpi_rank == 0:
    stop = 0
    x = scipy.optimize.minimize(
        objective, 
        x0,
        method="trust-constr", 
        args=(stop), 
        bounds=scipy.optimize.Bounds(lb, ub),
        options=dict(verbose=2)
    )
    stop = 1
    objective(x0, stop)

else:
    stop = 0
    while stop == 0:
        objective(x0, stop)