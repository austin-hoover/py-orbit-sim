"""Backtrack from HZ04 to RFQ exit."""
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
from orbit.bunch_generators import TwissContainer
from orbit.bunch_generators import GaussDist3D
from orbit.bunch_generators import WaterBagDist3D
from orbit.diagnostics import diagnostics
from orbit.lattice import AccActionsContainer
from orbit.py_linac.lattice import LinacAccLattice
from orbit.py_linac.lattice import LinacAccNodes
from orbit.utils import consts
import orbit_mpi

from SNS_BTF import SNS_BTF

sys.path.append(os.getcwd())
from pyorbit_sim import utils
from pyorbit_sim.bunch_utils import gen_bunch
from pyorbit_sim.linac import Monitor
from pyorbit_sim.linac import track_bunch
from pyorbit_sim.linac import track_bunch_reverse
from pyorbit_sim.utils import ScriptManager


# Setup
# --------------------------------------------------------------------------------------

# MPI
_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

# Create output directory and save script info.
man = ScriptManager(datadir="/home/46h/sim_data/", path=pathlib.Path(__file__))
man.save_info()
man.save_script_copy()
print("Script info:")
pprint(man.get_info())


# Lattice
# ------------------------------------------------------------------------------
max_drift_length = 0.0025  # [m]
file_path = os.path.dirname(os.path.realpath(__file__))
btf = SNS_BTF(
    coef_filename=os.path.join(file_path, "data/magnets/default_i2gl_coeff.csv")
)
btf.init_lattice(
    xml=os.path.join(file_path, "data/xml/BTF_lattice_default.xml"),
    beamlines=["MEBT1"], 
    max_drift_length=max_drift_length,
)
btf.update_quads_from_mstate(
    os.path.join(file_path, "data/mstate/TransmissionBS34_04212022.mstate"),
    value_type="current",
)
btf.add_aperture_nodes(drift_step=0.1)
btf.add_space_charge_nodes(
    grid_size_x=64, 
    grid_size_y=64,
    grid_size_z=64,
    path_length_min=max_drift_length,
    n_bunches=3,
    freq=402.5e6,
)
lattice = btf.lattice

# Add diagnostics (dump bunch nodes)
diag_parent_nodes = []
for diag_parent_node in diag_parent_nodes:
    filename = man.get_filename("bunch_{}.dat".format(diag_parent_node.getName()))
    diag_parent_node.addChildNode(
        diagnostics.DumpBunchNode(filename, verbose=True), 
        diag_parent_node.ENTRANCE, 
        part_index=0,
        place_in_part=AccActionsContainer.BEFORE,
    )

# Save node positions.
if _mpi_rank == 0:
    file = open(man.get_filename("nodes.dat"), "w")
    file.write("node, position\n")
    for node in lattice.getNodes():
        file.write("{}, {}, {}\n".format(node.getName(), node.getPosition(), node.getLength()))
    file.close()

    # Write lattice structure to file.
    file = open(man.get_filename("lattice_structure.txt"), "w")
    file.write(lattice.structureToText())
    file.close()

    
# Bunch
# ------------------------------------------------------------------------------

filename = os.path.join(
    "/home/46h/projects/BTF/meas_analysis/2022-06-26_scan-xxpy-image-ypdE/data/",
    "220626140058-scan-xxpy-image-ypdE_samp6D_1.00e+07.dat",
)
bunch = Bunch()
if _mpi_rank == 0:
    print("Reading bunch from file '{}'.".format(filename))
bunch.readBunch(filename)
bunch.mass(0.939294)  # [GeV / c^2]
bunch.charge(-1.0)  # [elementary charge units]
bunch.getSyncParticle().kinEnergy(0.0025)  # [GeV]
bunch_current = 0.0255  # [A]
bunch_freq = 402.5e6  # [Hz]
bunch_charge = bunch_current / bunch_freq
intensity = bunch_charge / abs(float(bunch.charge()) * consts.charge_electron)
bunch_size_global = bunch.getSizeGlobal()
bunch.macroSize(intensity / bunch_size_global)

# If `dist` is not None, generate an RMS-equivalent distribution in x-x',
# y-y', and z-z' using an analytic distribution function (such as Gaussian, 
# KV, or Waterbag). Then reconstruct the the six-dimensional distribution as
# f(x, x', y, y', z, z') = f(x, x') f(y, y') f(z, z').
dist = None
if dist is not None:
    if _mpi_rank == 0:
        print("Repopulating bunch using 2D Twiss parameters and {} generator.".format(dist))
    bunch_twiss_analysis = BunchTwissAnalysis()
    dispersion_flag = 0
    emit_norm_flag = 0
    order = 2
    bunch_twiss_analysis.computeBunchMoments(bunch, order, dispersion_flag, emit_norm_flag)    
    eps_x = bunch_twiss_analysis.getEffectiveEmittance(0)
    eps_y = bunch_twiss_analysis.getEffectiveEmittance(1)
    eps_z = bunch_twiss_analysis.getEffectiveEmittance(2)
    beta_x = bunch_twiss_analysis.getEffectiveBeta(0)
    beta_y = bunch_twiss_analysis.getEffectiveBeta(1)
    beta_z = bunch_twiss_analysis.getEffectiveBeta(2)
    alpha_x = bunch_twiss_analysis.getEffectiveAlpha(0)
    alpha_y = bunch_twiss_analysis.getEffectiveAlpha(1)
    alpha_z = bunch_twiss_analysis.getEffectiveAlpha(2)
    dist = dist(
        twissX=TwissContainer(alpha_x, beta_x, eps_x),
        twissY=TwissContainer(alpha_y, beta_y, eps_y),
        twissZ=TwissContainer(alpha_z, beta_z, eps_z),
    )
    bunch = gen_bunch(dist=dist, n_parts=bunch_size_global, bunch=bunch, verbose=True)
    
if _mpi_rank == 0:
    print("Bunch parameters:")
    print("  charge = {}".format(bunch.charge()))
    print("  mass = {} [GeV / c^2]".format(bunch.mass()))
    print("  kinetic energy = {} [GeV]".format(bunch.getSyncParticle().kinEnergy()))
    print("  macrosize = {}".format(bunch.macroSize()))
    print("  size (local) = {:.2e}".format(bunch.getSize()))
    print("  size (global) = {:.2e}".format(bunch_size_global))
    
    
# Sim
# ------------------------------------------------------------------------------

start = None  # stop node in reversed lattice
stop = "MEBT:HZ04"  # start node in reversed lattice

monitor = Monitor(
    start_position=0.0, # this will be set automatically in `track_bunch`.
    plotter=None,
    verbose=True,
    track_history=True,
    track_rms=True,
    dispersion_flag=True,
    emit_norm_flag=False,
)

if _mpi_rank == 0:
    print("Tracking...")
track_bunch_reverse(bunch, lattice, monitor=monitor, start=start, stop=stop, verbose=True)

filename = man.get_filename("bunch_RFQ.dat")
if _mpi_rank == 0:
    print("Saving bunch to file {}".format(filename))
bunch.dumpBunch(filename)
    
if _mpi_rank == 0 and monitor.track_history:
    filename = man.get_filename("history.dat")
    print("Writing history to {}".format(filename))
    monitor.write(filename, delimiter=",")