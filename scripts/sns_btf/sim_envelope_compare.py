"""Compare RMS-equivalent KV distribution with bunch."""
from __future__ import print_function
import os
import pathlib
from pprint import pprint
import sys
import time

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

import pyorbit_sim

from sns_btf import SNS_BTF


# Setup
# --------------------------------------------------------------------------------------

# Set up directories.
output_dir = "/home/46h/sim_data/"  # parent directory for output
file_dir = os.path.dirname(os.path.realpath(__file__))  # directory of this file
input_dir = os.path.join(file_dir, "data_input")  # lattice input data
    

# Bunch
# --------------------------------------------------------------------------------------

# Settings
filename = os.path.join(
    "/home/46h/projects/BTF/sim/SNS_RFQ/parmteq/2021-01-01_benchmark/data/",
    "bunch_RFQ_output_1.00e+05.dat",
)
mass = 0.939294  # [GeV / c^2]
charge = -1.0  # [elementary charge units]
kin_energy = 0.0025  # [GeV]
current = 0.042  # [A]
intensity = pyorbit_sim.bunch_utils.get_intensity(current, 402.5e+06)
intensity = 0.0
n_parts = int(1.00e+05)  # max number of particles (just need correct rms)

# Generate bunch.
bunch = Bunch()
bunch.mass(mass)
bunch.charge(charge)
bunch.getSyncParticle().kinEnergy(kin_energy)
bunch = pyorbit_sim.bunch_utils.load(filename=filename, bunch=bunch)
bunch = pyorbit_sim.bunch_utils.downsample(bunch, n=n_parts, verbose=True)
bunch = pyorbit_sim.bunch_utils.set_centroid(bunch, centroid=0.0)
bunch_size_global = bunch.getSizeGlobal()
macro_size = intensity / bunch_size_global
bunch.macroSize(macro_size)
params_dict = {"bunch": bunch}

# Generate rms-equivalent KV distribution envelope.
cov, mean = pyorbit_sim.bunch_utils.get_stats(bunch)
bunch_length_rms = 2.0 * np.sqrt(cov[4, 4])
bunch_length = 1.0 * bunch_length_rms
envelope = DanilovEnvelope20(
    mass=mass,
    kin_energy=kin_energy,
    length=bunch_length,
    intensity=intensity,
)
envelope.set_cov(cov)

print("Created rms-equivalent KV distribution envelope")
print("Envelope twiss:", envelope.twiss())
print("Bunch twiss:   ", pyorbit_sim.stats.twiss(cov[:4, :4]))


# Lattice
# --------------------------------------------------------------------------------------

def get_linear_linac():
    linac = SNS_BTF(
        coef_filename=os.path.join(input_dir, "magnets/default_i2gl_coeff.csv"),
        rf_frequency=402.5e06,
    )
    linac.init_lattice(
        xml_filename=os.path.join(input_dir, "xml/btf_lattice_straight.xml"),
        sequences=["MEBT1", "MEBT2"],
        max_drift_length=0.010,
    )
    linac.set_fringe_fields(False)
    linac.set_linac_tracker(False)
    return linac


# Create lattice for envelope.
linac_env = get_linear_linac()
linac_env.add_envelope_solver_nodes_2d(
    path_length_min=0.010,
    perveance=envelope.perveance,
    eps_x=envelope.eps_x,
    eps_y=envelope.eps_y,
)
lattice_env = linac_env.lattice

# Create lattice for bunch.
linac = get_linear_linac()
linac.add_space_charge_nodes(
    grid_size_x=64,
    grid_size_y=64,
    grid_size_z=64,
    path_length_min=0.010,
    n_bunches=1,
)
lattice = linac.lattice

index_stop = lattice.getNodeIndex(lattice.getNodeForName("MEBT:VS06"))


# Envelope tracking
# --------------------------------------------------------------------------------------

class Monitor:
    def __init__(self, verbose=True, stride=0.100):
        self.history = []
        self.verbose = verbose
        self.stride = stride
        self.position = 0.0
        self.flag = False
        
    def get_stats(self, bunch):
        return
        
    def action(self, params_dict):
        position = params_dict["path_length"]
        if self.flag and (position - self.position < self.stride):
            return
        self.position = position
        self.flag = True

        bunch = params_dict["bunch"]
        x_rms, y_rms = self.get_stats(bunch)
        self.history.append([position, x_rms, y_rms])
        if self.verbose:
            print("{:.5f} {:.3f} {:.3f}".format(position, x_rms, y_rms))


class EnvelopeMonitor(Monitor):
    def __init__(self, verbose=True, stride=0.100):
        Monitor.__init__(self, verbose=verbose, stride=stride)
        
    def get_stats(self, bunch):
        x_rms = 0.5 * 1000.0 * bunch.x(0)
        y_rms = 0.5 * 1000.0 * bunch.y(0)
        return (x_rms, y_rms)
    
    
print("Tracking envelope")
env_bunch, env_params_dict = envelope.to_bunch()
env_monitor = EnvelopeMonitor(verbose=True, stride=0.100)
action_container = AccActionsContainer()
action_container.addAction(env_monitor.action, AccActionsContainer.ENTRANCE)
action_container.addAction(env_monitor.action, AccActionsContainer.EXIT)
lattice_env.trackBunch(
    env_bunch,
    paramsDict=env_params_dict,
    actionContainer=action_container,
    index_start=0,
    index_stop=index_stop,
)
    
    
    
# Bunch tracking
# --------------------------------------------------------------------------------------

class BunchMonitor(Monitor):
    def __init__(self, verbose=True, stride=0.100):
        Monitor.__init__(self, verbose=verbose, stride=stride)
        
    def get_stats(self, bunch):
        position = params_dict["path_length"]
        bunch = params_dict["bunch"]
        twiss_analysis = BunchTwissAnalysis()
        order = 2
        twiss_analysis.computeBunchMoments(bunch, order, 0, 0)
        sig_xx = twiss_analysis.getCorrelation(0, 0)
        sig_yy = twiss_analysis.getCorrelation(2, 2)
        x_rms = 1000.0 * np.sqrt(sig_xx)
        y_rms = 1000.0 * np.sqrt(sig_yy)
        return (x_rms, y_rms)
    
    
print("Tracking bunch")
bunch_monitor = BunchMonitor(verbose=True, stride=0.100)
action_container = AccActionsContainer()
action_container.addAction(bunch_monitor.action, AccActionsContainer.ENTRANCE)
action_container.addAction(bunch_monitor.action, AccActionsContainer.EXIT)
lattice.trackBunch(
    bunch,
    paramsDict=params_dict,
    actionContainer=action_container,
    index_start=0,
    index_stop=index_stop,
)
    
print("Comparison")
print("(env) s, (bunch) s, (env) xrms, (bunch) xrms, (env) yrms, (bunch) yrms")
for (s1, x1, y1), (s2, x2, y2) in zip(env_monitor.history, bunch_monitor.history):
    print(
        "{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
            s1, s2, x1, x2, y1, y2
        )
    )