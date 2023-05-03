"""SNS Linac simulation."""
from __future__ import print_function
import math
import os
import pathlib
import pickle
from pprint import pprint
import random
import shutil
import sys
import time

import numpy as np

from bunch import Bunch
from bunch import BunchTwissAnalysis
from bunch_utils_functions import copyCoordsToInitCoordsAttr
from linac import BaseRfGap
from linac import BaseRfGap_slow
from linac import MatrixRfGap
from linac import RfGapThreePointTTF_slow
from linac import RfGapTTF
from linac import RfGapTTF_slow
from orbit.bunch_generators import GaussDist3D
from orbit.bunch_generators import KVDist3D
from orbit.bunch_generators import TwissContainer
from orbit.bunch_generators import WaterBagDist3D
from orbit.bunch_utils import ParticleIdNumber
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.py_linac.lattice_modifications import Add_quad_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_rfgap_apertures_to_lattice
from orbit.py_linac.lattice_modifications import AddMEBTChopperPlatesAperturesToSNS_Lattice
from orbit.py_linac.lattice_modifications import AddScrapersAperturesToLattice
from orbit.py_linac.lattice_modifications import GetLostDistributionArr
from orbit.py_linac.lattice_modifications import Replace_BaseRF_Gap_and_Quads_to_Overlapping_Nodes
from orbit.py_linac.lattice_modifications import Replace_BaseRF_Gap_to_AxisField_Nodes
from orbit.py_linac.lattice_modifications import Replace_Quads_to_OverlappingQuads_Nodes
from orbit.py_linac.linac_parsers import SNS_LinacLatticeFactory
from orbit.py_linac.overlapping_fields import SNS_EngeFunctionFactory
from orbit.space_charge.sc3d import setSC3DAccNodes
from orbit.space_charge.sc3d import setUniformEllipsesSCAccNodes
from orbit.space_charge.sc2p5d import setSC2p5DrbAccNodes
from orbit.utils import consts
import orbit_mpi
from spacecharge import SpaceChargeCalc3D
from spacecharge import SpaceChargeCalcUnifEllipse

from SNS_LINAC import SNS_LINAC

sys.path.append(os.getcwd())
import pyorbit_sim.bunch_utils
import pyorbit_sim.linac
import pyorbit_sim.plotting
from pyorbit_sim.misc import lorentz_factors
from pyorbit_sim.utils import ScriptManager


# Setup
# --------------------------------------------------------------------------------------

save = True  # no output if false

# MPI
_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

# Create output directory and save script info.
man = ScriptManager(datadir="/home/46h/sim_data/", path=pathlib.Path(__file__))
if save:
    man.save_info()
    man.save_script_copy()
print("Script info:")
pprint(man.get_info())

file_path = os.path.dirname(os.path.realpath(__file__))


# Lattice
# --------------------------------------------------------------------------------------

xml_file_name = os.path.join(file_path, "data/sns_linac.xml")
max_drift_length = 0.01  # [m]
sequences = [
    "MEBT",
    "DTL1",
    "DTL2",
    "DTL3",
    "DTL4",
    "DTL5",
    "DTL6",
    "CCL1",
    "CCL2",
    "CCL3",
    "CCL4",
    "SCLMed",
    "SCLHigh",
    "HEBT1",
    "HEBT2",
]
linac = SNS_LINAC(xml=xml_file_name, max_drift_length=max_drift_length, sequences=sequences, verbose=True)
lattice = linac.lattice

# Save node positions.
if save:
    file = open(man.get_filename("nodes.txt"), "w")
    for node in lattice.getNodes():
        file.write("{} {}\n".format(node.getName(), node.getPosition()))
    file.close()

## Set the RF gap model.
linac.set_rf_gap_model(RfGapTTF)

## Set overlapping RF and quad fields.
fields_filename = os.path.join(file_path, "data/fields.xml")
z_step = 0.002
# [...]

## Add space charge nodes.
sc_nodes = linac.add_space_charge_nodes(
    solver="3D",  # {"3D", "ellipsoid"}
    grid_size=(64, 64, 64), 
    path_length_min=0.01,
    n_bunches=1,
    freq=402.5e-6,
    verbose=True,
)

## Add aperture nodes.
aperture_nodes = linac.add_aperture_nodes(x_size=0.042, y_size=0.042, verbose=True)

## Use linac-style quads and drifts instead of TEAPOT style. (Useful when 
## the energy spread is large, but is slower and is not symplectic.)
linac_tracker = True
lattice.setLinacTracker(linac_tracker)
print("Set linac tracker {}".format(linac_tracker))


# Bunch (generate from 2D Twiss parameters)
# --------------------------------------------------------------------------------------

# Unnormalized transverse emittances; units = [pi * mm * mrad].
# Longitudinal emittance; units [eV * sec].
kin_energy = 0.0025  # [GeV]
mass = 0.939294  # [GeV / c^2]
gamma, beta = lorentz_factors(mass=mass, kin_energy=kin_energy)

# Emittances are normalized - transverse by gamma * beta, longitudinal by gamma**3 * beta.
(alpha_x, beta_x, eps_x) = (-1.9620, 0.1831, 0.21)
(alpha_y, beta_y, eps_y) = (1.7681, 0.1620, 0.21)
(alpha_z, beta_z, eps_z) = (0.0196, 0.5844, 0.24153)

alpha_z = -alpha_z

# Make emittances un-normalized XAL units [m*rad].
eps_x = 1.0e-6 * eps_x / (gamma * beta)
eps_y = 1.0e-6 * eps_y / (gamma * beta)
eps_z = 1.0e-6 * eps_z / (gamma**3 * beta)

# Convert longitudinal emittance units to [GeV * m].
eps_z = eps_z * gamma**3 * beta**2 * mass
beta_z = beta_z / (gamma**3 * beta**2 * mass)

print("Generating bunch from Twiss parameters.")
bunch = pyorbit_sim.bunch_utils.generate(
    dist=WaterBagDist3D(
        twissX=TwissContainer(alpha_x, beta_x, eps_x),
        twissY=TwissContainer(alpha_y, beta_y, eps_y),
        twissZ=TwissContainer(alpha_z, beta_z, eps_z),
    ), 
    n_parts=int(1.0e4), 
    verbose=True,
)

# Set beam parameters.
bunch.mass(mass)  # [GeV / c^2]
bunch.charge(-1.0)  # [elementary charge units]
bunch.getSyncParticle().kinEnergy(kin_energy)  # [GeV]
current = 0.038  # [A]
bunch_frequency = 402.5e6  # [Hz]
bunch_charge = current / bunch_frequency
intensity = bunch_charge / abs(float(bunch.charge()) * consts.charge_electron)
bunch_size_global = bunch.getSizeGlobal()
bunch.macroSize(intensity / bunch_size_global)

if _mpi_rank == 0:
    print("Bunch parameters:")
    print("  charge = {}".format(bunch.charge()))
    print("  mass = {} [GeV / c^2]".format(bunch.mass()))
    print("  kinetic energy = {} [GeV]".format(bunch.getSyncParticle().kinEnergy()))
    print("  macrosize = {}".format(bunch.macroSize()))
    print("  size (local) = {:.2e}".format(bunch.getSize()))
    print("  size (global) = {:.2e}".format(bunch_size_global))
    
## Assign ID number to each particle.
# ParticleIdNumber.addParticleIdNumbers(bunch)
# copyCoordsToInitCoordsAttr(bunch)


# Tracking
# --------------------------------------------------------------------------------------

start = 0  # start node (name or position)
# stop = "SCL_Diag:BPM24"  # stop node (name or position)s
stop = 30.0
save_input_bunch = True
save_output_bunch = True

writer = pyorbit_sim.linac.BunchWriter(
    folder=man.outdir, 
    prefix=man.prefix, 
    index=1, 
)


def transform(X):
    """Normalize the 2D phase spaces."""
    Sigma = np.cov(X.T)
    Xn = np.zeros(X.shape)
    for i in range(0, X.shape[1], 2):
        sigma = Sigma[i : i + 2, i : i + 2]
        eps = np.sqrt(np.linalg.det(sigma))
        alpha = -sigma[0, 1] / eps
        beta = sigma[0, 0] / eps
        Xn[:, i] = X[:, i] / np.sqrt(beta)
        Xn[:, i + 1] = (np.sqrt(beta) * X[:, i + 1]) + (alpha * X[:, i] / np.sqrt(beta))
        Xn[:, i : i + 2] = Xn[:, i : i + 2] / np.sqrt(eps)
    return Xn
    
plotter = pyorbit_sim.plotting.Plotter(
    transform=transform, 
    folder=man.outdir,
    default_save_kws=None, 
)
plotter.add_function(
    pyorbit_sim.plotting.proj2d_three_column, 
    save_kws=None, 
    name=None, 
    bins=32,
)

monitor = pyorbit_sim.linac.Monitor(
    position_offset=0.0,  # will be set automatically in `track`.
    stride={
        "update": 0.1,  # [m]
        "write_bunch": None,  # [m]
        "plot_bunch": 5.0,  # [m]
    },
    writer=writer,
    plotter=plotter,
    track_history=True,
    track_rms=True,
    dispersion_flag=False,
    emit_norm_flag=False,
    verbose=True,
)

# Record synchronous particle time of arrival at each accelerating cavity.
if _mpi_rank == 0:
    print("Tracking design bunch...")
lattice.trackDesignBunch(bunch)
if _mpi_rank == 0:
    print("Design bunch tracking complete.")
    
# Save input bunch.
if save_input_bunch and save:
    if start is None or type(start) is not str:
        filename = man.get_filename("bunch_0_START.dat")
    else:
        filename = man.get_filename("bunch_0_{}.dat".format(start))
    if _mpi_rank == 0:
        print("Saving bunch to file {}".format(filename))
    bunch.dumpBunch(filename)    
    
# Track
if _mpi_rank == 0:
    print("Tracking...")
        
params_dict = pyorbit_sim.linac.track(
    bunch, 
    lattice, 
    monitor=monitor, 
    start=start, 
    stop=stop, 
    verbose=True
)

# Save history.
if _mpi_rank == 0 and monitor.track_history and save:
    filename = man.get_filename("history.dat")
    print("Writing history to {}".format(filename))
    monitor.write_history(filename, delimiter=",")
    
# Save lost particles.
lostbunch = params_dict["lostbunch"]
aprt_nodes_losses = GetLostDistributionArr(aperture_nodes, lostbunch)
total_loss = 0.0
for (node, loss) in aprt_nodes_losses:
    print(
        "node={:30s},".format(node.getName()), 
        "pos={:9.3f},".format(node.getPosition()), 
        "loss= {:6.0f}".format(loss),
    )
    total_loss += loss
print("Total loss = {:.2e}".format(total_loss))
if save:
    filename = man.get_filename("losses.txt")
    print("Saving loss vs. node array to {}".format(filename))
    file = open(filename, "w")
    for (node, loss) in aprt_nodes_losses:
        file.write("{} {}\n".format(node.getName(), loss))
    file.close()

    filename = man.get_filename("bunch_lost.dat".format(stop))
    if _mpi_rank == 0:
        print("Saving lost bunch to file {}".format(filename))
    bunch.dumpBunch(filename)    
    
# Save output bunch.
if save_output_bunch and save:
    if stop is None or type(stop) is not str:
        filename = man.get_filename("bunch_{}_STOP.dat".format(writer.index))
    else:
        filename = man.get_filename("bunch_{}_{}.dat".format(writer.index, stop))
    if _mpi_rank == 0:
        print("Saving bunch to file {}".format(filename))
    bunch.dumpBunch(filename)

print(man.timestamp)