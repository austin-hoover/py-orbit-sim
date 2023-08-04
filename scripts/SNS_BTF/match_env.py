"""Match the beam to the FODO line."""
from __future__ import print_function
import os
from pprint import pprint
import sys

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize


from bunch import Bunch
from bunch import BunchTwissAnalysis
from orbit.envelope import DanilovEnvelope20
from orbit.envelope import DanilovEnvelopeSolverNode20
from orbit.envelope import set_danilov_envelope_solver_nodes_20
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.py_linac.lattice import LinacAccLattice
from orbit.py_linac.lattice import OverlappingQuadsNode
from orbit.py_linac.lattice import Quad
import orbit_mpi

sys.path.append("../../")
import pyorbit_sim

from SNS_BTF import SNS_BTF


# Setup
# --------------------------------------------------------------------------------------

# Create the lattice.
linac = SNS_BTF(
    coef_filename="data_input/magnets/default_i2gl_coeff.csv",
    rf_frequency=402.5e06,
)
lattice = linac.init_lattice(
    xml_filename="data_input/xml/btf_lattice_straight.xml",
    sequences=["MEBT1", "MEBT2"],
    max_drift_length=0.010,
)
lattice.setLinacTracker(False)
node_positions_dict = lattice.getNodePositionsDict()

# Initial bunch settings
filename = os.path.join(
    "/home/46h/projects/BTF/sim/SNS_RFQ/parmteq/2021-01-01_benchmark/data/",
    "bunch_RFQ_output_1.00e+05.dat",
)
mass = 0.939294  # [GeV / c^2]
charge = -1.0  # [elementary charge units]
kin_energy = 0.0025  # [GeV]
current = 0.040  # [A]
intensity = pyorbit_sim.bunch_utils.get_intensity(current, linac.rf_frequency)
n_parts = int(5.0e04)  # max number of particles

# Load initial bunch.
bunch = Bunch()
bunch.mass(mass)
bunch.charge(charge)
bunch.getSyncParticle().kinEnergy(kin_energy)
bunch = pyorbit_sim.bunch_utils.load(
    filename=filename,
    bunch=bunch,
)
bunch = pyorbit_sim.bunch_utils.downsample(bunch, n=n_parts, verbose=True)
bunch = pyorbit_sim.bunch_utils.set_centroid(bunch, centroid=0.0)
bunch_size_global = bunch.getSizeGlobal()
macro_size = intensity / bunch_size_global
bunch.macroSize(macro_size)

# Track the bunch to the matching section.
print("Tracking to MEBT:VS06")
index_start = 0
index_stop = lattice.getNodeIndex(lattice.getNodeForName("MEBT:VS06"))
lattice.trackBunch(bunch, index_start=index_start, index_stop=index_stop)


# Envelope matching
# --------------------------------------------------------------------------------------

print("Computing matched envelope...")

# Generate rms-equivalent KV distribution envelope.
cov, mean = pyorbit_sim.bunch_utils.get_stats(bunch)
cx = 2.0 * np.sqrt(cov[0, 0])
cy = 2.0 * np.sqrt(cov[2, 2])
cxp = 4.0 * cov[0, 1] / cx
cyp = 4.0 * cov[2, 3] / cy
eps_x = np.sqrt(np.linalg.det(cov[0:2, 0:2]))
eps_y = np.sqrt(np.linalg.det(cov[2:4, 2:4]))
bunch_length = 2.0 * (2.0 * np.sqrt(cov[4, 4]))
envelope = DanilovEnvelope20(
    eps_x=eps_x,
    eps_y=eps_y,
    mass=mass,
    kin_energy=kin_energy,
    length=bunch_length,
    intensity=intensity,
    params=[cx, cxp, cy, cyp],
)
print("Created envelope")
print("Envelope twiss:", envelope.twiss())
print("Bunch twiss:   ", pyorbit_sim.stats.twiss(cov[:4, :4]))

# Save envelope parameters.
envelope_params_init = np.copy(envelope.params)

# Add envelope solver nodes to the lattice.
solver_nodes = set_danilov_envelope_solver_nodes_20(
    lattice,
    path_length_min=0.001,
    path_length_max=0.005,
    perveance=envelope.perveance,
    eps_x=envelope.eps_x,
    eps_y=envelope.eps_y,
)

# Turn off nonlinear quadrupole fringe fields.
for node in lattice.getNodes():
    try:
        node.setUsageFringeFieldIN(False)
        node.setUsageFringeFieldOUT(False)
    except:
        pass

# Update start/stop node indices to beginning/end of FODO line.
index_start = index_stop + 1
index_stop = lattice.getNodeIndex(lattice.getNodeForName("MEBT:QH30"))


class OpticsController:
    def __init__(self, lattice=None, quad_names=None):
        self.lattice = lattice
        self.quad_names = quad_names
        self.quad_nodes = [lattice.getNodeForName(name) for name in quad_names]

    def get_quad_strengths(self):
        return [node.getParam("dB/dr") for node in self.quad_nodes]

    def set_quad_strengths(self, x):
        for i, node in enumerate(self.quad_nodes):
            node.setParam("dB/dr", x[i])

    def get_quad_bounds(self, factor=0.5):
        """Return (lower_bounds, upper_bounds) for quad strengths."""
        bounds = []
        for kq in self.get_quad_strengths():
            lb = (1.0 - factor) * kq
            ub = (1.0 + factor) * kq
            if ub < lb:
                (lb, ub) = (ub, lb)
            bounds.append([lb, ub])
        return np.array(bounds).T
    
        
class Monitor:
    def __init__(self, node_names=None, position_offset=0.0):
        self.node_names = node_names
        self.position_offset = position_offset
        self.history = []

    def reset(self):
        self.history = []

    def action(self, params_dict):
        bunch = params_dict["bunch"]
        node = params_dict["node"]
        position = params_dict["path_length"] + self.position_offset
        if self.node_names and (node.getName() not in self.node_names):
            return
        cx = 1000.0 * bunch.x(0)
        cy = 1000.0 * bunch.y(0)
        self.history.append([position, cx, cy])
        
        
class Optimizer:
    def __init__(
        self,
        lattice=None,
        optics_controller=None,
        fodo_quad_names=None,
        save_freq=None,
        index_start=0,
        index_stop=-1,
    ):
        self.lattice = lattice
        self.optics_controller = optics_controller
        self.fodo_quad_names = fodo_quad_names
        self.count = 0
        self.ymax = None
        self.save_freq = save_freq
        self.index_start = index_start
        self.index_stop = index_stop
        self.position_offset, _ = node_positions_dict[lattice.getNodes()[index_start]]

    def track_envelope(self, dense=False):
        """Return (x_rms, y_rms) at each FODO quad."""
        envelope.set_params(envelope_params_init)
        _bunch, _params_dict = envelope.to_bunch()
        
        monitor_node_names = None if dense else fodo_quad_names
        monitor = Monitor(node_names=monitor_node_names, position_offset=self.position_offset)
        action_container = AccActionsContainer()
        action_container.addAction(monitor.action, AccActionsContainer.ENTRANCE)
        
        self.lattice.trackBunch(
            _bunch,
            paramsDict=_params_dict,
            actionContainer=action_container,
            index_start=self.index_start,
            index_stop=self.index_stop,
        )
        return np.array(monitor.history)

    def save_data(self):
        # Save quadrupole strengths.
        filename = "./data_output/quad_strengths_{:06.0f}.dat".format(self.count)
        file = open(filename, "w")
        file.write("quad_name dB/dr\n")
        for node in self.lattice.getNodesOfClasses([Quad, OverlappingQuadsNode]):
            file.write("{} {}\n".format(node.getName(), node.getParam("dB/dr")))
        file.close()
        
        # Save envelope trajectory.
        history = self.track_envelope(dense=True)       
        history = pd.DataFrame(history, columns=["position", "x_rms", "y_rms"])
        history["x_rms"] = 0.5 * history["x_rms"]
        history["y_rms"] = 0.5 * history["y_rms"]
        filename = "./data_output/envelope_trajectory_{:06.0f}.dat".format(self.count)
        history.to_csv(filename, sep=" ")
        
        # Plot beam size evolution.
        positions = history.loc[:, "position"].values
        x_rms = history.loc[:, "x_rms"].values
        y_rms = history.loc[:, "y_rms"].values
        
        fig, ax = plt.subplots(figsize=(7.0, 2.5), tight_layout=True)
        kws = dict()
        ax.plot(positions, x_rms, label="x", **kws)
        ax.plot(positions, y_rms, label="y", **kws)
        if self.count == 0:
            self.ymax = ax.get_ylim()[1]
        ax.set_ylim((0.0, self.ymax))
        
        node_start = self.lattice.getNodes()[self.index_start]
        for node in optics_controller.quad_nodes:
            start, stop = node_positions_dict[node]
            ax.axvspan(start, stop, color="black", alpha=0.075, ec="None")
          
        ax.set_xlabel("Position [m]")
        ax.set_ylabel("RMS size [mm]")
        ax.legend(loc="upper right")
        filename = "./data_output/envelope_{:06.0f}.png".format(self.count)
        plt.savefig(filename, dpi=100)
        plt.close()

    def objective(self, x):
        self.optics_controller.set_quad_strengths(x)
        history = self.track_envelope(dense=False)
        
        cost = 0.0        
        cx = history[:, 1]
        cy = history[:, 2]
        for i in range(2):
            cost += np.var(cx[i::2])
            cost += np.var(cy[i::2])

        if self.save_freq and (self.count % self.save_freq == 0):
            self.save_data()
            
        self.count += 1
        return cost


fodo_quad_names = ["MEBT:FQ{}".format(i) for i in range(11, 34)]
matching_quad_names = [
    "MEBT:QV07",
    "MEBT:QH08",
    "MEBT:QV09",
    "MEBT:QH10",
]
optics_controller = OpticsController(lattice, matching_quad_names)

optimizer = Optimizer(
    lattice, 
    optics_controller=optics_controller, 
    fodo_quad_names=fodo_quad_names, 
    save_freq=300,
    index_start=index_start,
    index_stop=index_stop,
)
objective = optimizer.objective

x0 = optics_controller.get_quad_strengths()
bounds = optics_controller.get_quad_bounds()
result = scipy.optimize.least_squares(
    objective,
    x0,
    bounds=bounds,
    verbose=2,
    max_nfev=10000,
)
optimizer.save_data()
