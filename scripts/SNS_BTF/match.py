"""Match the beam to the FODO line."""
from __future__ import print_function
import os
import sys

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize


from bunch import Bunch
from bunch import BunchTwissAnalysis
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
import orbit_mpi

sys.path.append("../../")
import pyorbit_sim.bunch_utils
import pyorbit_sim.linac

from SNS_BTF import SNS_BTF





# Setup
# --------------------------------------------------------------------------------------

# Create the lattice.
linac = SNS_BTF(
    coef_filename="data_input/magnets/default_i2gl_coeff.csv",
    rf_frequency=402.5e+06,
)
linac.init_lattice(
    xml_filename="data_input/xml/btf_lattice_straight.xml",
    sequences=["MEBT1", "MEBT2"],
    max_drift_length=0.010,
)    
linac.add_uniform_ellipsoid_space_charge_nodes(
    n_ellipsoids=4,
    path_length_min=0.010,
)
linac.add_aperture_nodes(drift_step=0.1, verbose=True)
linac.set_linac_tracker(True)
lattice = linac.lattice
    

# Load the initial bunch.
filename = os.path.join(
    "/home/46h/projects/BTF/sim/SNS_RFQ/parmteq/2021-01-01_benchmark/data/",
    "bunch_RFQ_output_1.00e+05.dat",
)
mass = 0.939294  # [GeV / c^2]
charge = -1.0  # [elementary charge units]
kin_energy = 0.0025  # [GeV]
current = 0.040  # [A]
intensity = pyorbit_sim.bunch_utils.get_intensity(current, linac.rf_frequency)
n_parts = int(2.0e+04)  # max number of particles

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
index_stop = lattice.getNodeIndex(lattice.getNodeForName("MEBT:VS06"))
action_container = AccActionsContainer()
action_container.addAction(
    lambda params_dict: print("s={}".format(params_dict["path_length"])),
    AccActionsContainer.EXIT
)
lattice.trackBunch(
    bunch,
    actionContainer=action_container,
    index_start=0,
    index_stop=index_stop,
)


# Matching
# --------------------------------------------------------------------------------------

print("Matching...")

index_start = index_stop
index_stop = lattice.getNodeIndex(lattice.getNodeForName("MEBT:QH30"))
                                  

class OpticsController:
    def __init__(self, lattice=None, quad_names=None):
        self.lattice = lattice
        self.quad_names = quad_names
        self.quads = [lattice.getNodeForName(name) for name in quad_names]
        self.counter = 0
    
    def get_quad_strengths(self):
        return [quad.getParam("dB/dr") for quad in self.quads]
    
    def set_quad_strengths(self, x):
        for i, quad in enumerate(self.quads):
            quad.setParam("dB/dr", x[i])
        self.counter += 1
            
    def get_quad_bounds(self):
        # Estimate
        bounds = []
        for kq in self.get_quad_strengths():
            if kq == 0.0:
                kq = 6.0
            sign = np.sign(kq)
            lb = 0.0
            ub = 2.0 * kq
            if ub < lb:
                lb, ub = ub, lb
            bounds.append([lb, ub])
        return np.array(bounds).T
    

# Define matching quads.                            
quad_names = [
    "MEBT:QV07",
    "MEBT:QH08",
    "MEBT:QV09",
    "MEBT:QH10",
]
optics_controller = OpticsController(lattice=lattice, quad_names=quad_names)

    
class Monitor:
    # Track the beam size at each FODO quad.
    def __init__(self, node_names=None):
        self.history = []
        self.node_names = node_names
        if self.node_names is None:
            self.node_names = []
        
    def evaluate(self, bunch):
        twiss_analysis = BunchTwissAnalysis()
        order = 2
        dispersion_flag = 0
        emit_norm_flag = 0
        twiss_analysis.computeBunchMoments(bunch, order, dispersion_flag, emit_norm_flag)
        sig_x = 1000.0 * np.sqrt(twiss_analysis.getCorrelation(0, 0))
        sig_y = 1000.0 * np.sqrt(twiss_analysis.getCorrelation(2, 2))
        return sig_x * sig_y
        
    def action(self, params_dict):
        node = params_dict["node"]
        position = params_dict["path_length"]
        if node.getName() in self.node_names:
            bunch = params_dict["bunch"]
            metric = self.evaluate(bunch)
            self.history.append(metric)
            print("s={}, metric={}".format(position, metric))

            
fodo_quad_names = ["MEBT:FQ{}".format(i) for i in range(11, 34)]
counter = 0
        
def objective(x):
    # Set quad strengths (x), track, and report variance in rms beam area.
    optics_controller.set_quad_strengths(x)
    
    bunch_in = Bunch()
    bunch.copyBunchTo(bunch_in)
    
    monitor = Monitor(node_names=fodo_quad_names)
    action_container = AccActionsContainer()
    action_container.addAction(monitor.action, AccActionsContainer.EXIT)
    
    print("Tracking...")
    lattice.trackBunch(
        bunch_in,
        actionContainer=action_container,
        index_start=index_start,
        index_stop=index_stop,
    )
    metric_history = monitor.history
    metric_variance = np.var(metric_history)
    print("metric_variance = {}".format(metric_variance))
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(metric_history, color="black", marker=".")
    i = optics_controller.counter
    plt.savefig("data_output/figure_{:04.0f}.png".format(i))
    plt.close()
    
    file = open("data_output/test_quad_settings_{:04.0f}.txt".format(i), "w")
    for i in range(len(quad_names)):
        file.write("{} {}".format(quad_names[i], x[i]))
    file.close()
    
    return metric_variance


x0 = optics_controller.get_quad_strengths()
bounds = optics_controller.get_quad_bounds()
result = scipy.optimize.least_squares(
    objective, 
    x0, 
    bounds=bounds,
    verbose=2, 
)

file = open("data_output/test_quad_settings.txt", "w")
for i in range(len(result.x)):
    file.write("{} {}".format(quad_names[i], result.x[i]))
file.close()