"""Track a single particle through the SNS ring."""
from __future__ import print_function
import os
import pathlib
from pprint import pprint
import shutil
import sys
import time

import numpy as np
from tqdm import trange

from bunch import Bunch
from orbit.teapot import teapot
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_Ring
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice_Coupled
from orbit.utils import consts
import orbit_mpi

# sys.path.append("../..")
from pyorbit_sim.utils import ScriptManager

from SNS_RING import SNS_RING


# Setup
# --------------------------------------------------------------------------------------

save = True  # whether to save any data

# Set up directories.
file_dir = os.path.dirname(os.path.realpath(__file__))  # directory of this file
input_dir = os.path.join(file_dir, "data_input")  # lattice input data
output_dir = os.path.join(file_dir, "data_output")  # parent directory for output folder

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
    script_path_in_outdir=False,
)
if save and _mpi_rank == 0:
    man.make_outdir()
    man.save_info()
    man.save_script_copy()
    pprint(man.get_info())


# Lattice
# --------------------------------------------------------------------------------------

# Settings
madx_file = os.path.join(input_dir, "SNS_RING_nux6.18_nuy6.18_dual_solenoid/LATTICE.lat")
madx_seq = "rnginjsol"
mass = consts.mass_proton  # particle mass [GeV / c^2]
kin_energy = 0.800  # synchronous particle energy [GeV]

# Generate TEAPOT lattice.
ring = SNS_RING()
ring.readMADX(madx_file, madx_seq)
ring.set_fringe_fields(False)
ring.initialize()

# Set solenoid field strengths. The total integrated field should be 0.6 [T*m].
solenoid_names = ["scbdsol_c13a", "scbdsol_c13b"]
for name in solenoid_names:
    node = ring.getNodeForName(name)
    B = 0.6 / (2.0 * node.getLength())
    node.setParam("B", B)
    print("{}: B={:.2f}, L={:.2f}".format(node.getName(), node.getParam("B"), node.getLength()))

# Analyze the one-turn transfer matrix.
bunch = Bunch()
bunch.mass(mass)
bunch.getSyncParticle().kinEnergy(kin_energy)
matrix_lattice = TEAPOT_MATRIX_Lattice_Coupled(ring, bunch)
ring_params = matrix_lattice.getRingParametersDict()
pprint(ring_params)

# Add particles along each eigenvector.
eigenvectors = ring_params["eigenvectors"]
v1 = eigenvectors[:, 0]
v2 = eigenvectors[:, 2]
psi1 = 0.0
psi2 = 0.0
J1 = 50.0e-6
J2 = 50.0e-6
x1 = np.real(np.sqrt(2.0 * J1) * v1 * np.exp(-1.0j * psi1))
x2 = np.real(np.sqrt(2.0 * J2) * v2 * np.exp(-1.0j * psi2))
print("x1 =", x1)
print("x2 =", x2)

X = np.array([x1, x2])
bunch.deleteAllParticles()
for i in range(X.shape[0]):
    x, xp, y, yp = X[i, :]
    bunch.addParticle(x, xp, y, yp, 0.0, 0.0)
    
# Initialize TBT coordinate storage.
coords = []  # n_turns x n_parts x 6
for i in range(X.shape[0]):
    coords.append([X[i, :]])

# Track the bunch.
print()
print("Tracking bunch")
print("turn  | ip  | x        | x'       | y        | y'")
print("-------------------------------------------------------")
n_turns = 100
for turn in range(1, n_turns + 1):
    ring.trackBunch(bunch)
    for i in range(len(coords)):
        x = bunch.x(i)
        y = bunch.y(i)
        xp = bunch.xp(i)
        yp = bunch.yp(i)
        coords[i].append([x, xp, y, yp])
        print(
            "{:05.0f} | {:>3} | {:>8.3f} | {:>8.3f} | {:>8.3f} | {:>8.3f}".format(
                turn, 
                i, 
                1000.0 * x, 
                1000.0 * xp, 
                1000.0 * y, 
                1000.0 * yp,
            )
    )
        
# Save TBT coordinates.
for i in range(len(coords)):
    coords[i] = np.array(coords[i])
if save and _mpi_rank == 0:
    for i in range(len(coords)):
        mode = i + 1
        filename = man.get_filename("coords_mode{}.dat".format(mode))
        print("Saving file {}".format(filename))
        np.savetxt(filename, coords[i]) 

# Print timestamp again for convenience.
print("timestamp = {}".format(man.timestamp))
