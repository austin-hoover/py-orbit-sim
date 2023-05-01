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

from bunch import Bunch
from bunch import BunchTwissAnalysis
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
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.py_linac.lattice_modifications import Add_quad_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_rfgap_apertures_to_lattice
from orbit.py_linac.lattice_modifications import AddMEBTChopperPlatesAperturesToSNS_Lattice
from orbit.py_linac.lattice_modifications import AddScrapersAperturesToLattice
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
from pyorbit_sim import bunch_utils
from pyorbit_sim.linac import Monitor
from pyorbit_sim.linac import track_bunch
from pyorbit_sim.misc import lorentz_factors
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
bunch = bunch_utils.gen_bunch(
    dist=WaterBagDist3D(
        twissX=TwissContainer(alpha_x, beta_x, eps_x),
        twissY=TwissContainer(alpha_y, beta_y, eps_y),
        twissZ=TwissContainer(alpha_z, beta_z, eps_z),
    ), 
    n_parts=int(1e5), 
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


# Tracking
# --------------------------------------------------------------------------------------

start = 0  # start node (name or position)
stop = "SCL_Diag:BPM32"  # stop node (name or position)
save_input_bunch = False
save_output_bunch = False
monitor = Monitor(
    start_position=0.0,  # will be set automatically in `track_bunch`.
    plotter=None,
    verbose=True,
    track_history=True,
    track_rms=True,
    dispersion_flag=False,
    emit_norm_flag=False,
)

# Record synchronous particle time of arrival at each accelerating cavity.
if _mpi_rank == 0:
    print("Tracking design bunch...")
lattice.trackDesignBunch(bunch)
if _mpi_rank == 0:
    print("Design bunch tracking complete.")
    
# Save the input bunch.
if save_input_bunch:
    if start is None or start == 0:
        filename = man.get_filename("bunch_START.dat")
    else:
        filename = man.get_filename("bunch_{}.dat".format(start))
    if _mpi_rank == 0:
        print("Saving bunch to file {}".format(filename))
    bunch.dumpBunch(filename)    
    
# Track.
if _mpi_rank == 0:
    print("Tracking...")
track_bunch(bunch, lattice, monitor=monitor, start=start, stop=stop, verbose=True)

# Save history.
if _mpi_rank == 0 and monitor.track_history:
    filename = man.get_filename("history.dat")
    print("Writing history to {}".format(filename))
    monitor.write(filename, delimiter=",")

# Save the output bunch.
if save_output_bunch:
    if stop is None or stop == -1:
        filename = man.get_filename("bunch_STOP.dat")
    else:
        filename = man.get_filename("bunch_{}.dat".format(stop))
    if _mpi_rank == 0:
        print("Saving bunch to file {}".format(filename))
    bunch.dumpBunch(filename)
