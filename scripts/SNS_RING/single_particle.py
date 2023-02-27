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
from orbit.utils.consts import mass_proton

from SNS_RING import SNS_RING

sys.path.append(os.getcwd())
from pyorbit_sim import utils
from pyorbit_sim.utils import ScriptManager


# Setup
# --------------------------------------------------------------------------------------

# # Create output directory.
# outdir = "/home/46h/sim_data/"
# path = pathlib.Path(__file__)
# script_name = path.stem
# outdir = os.path.join(
#     outdir, 
#     path.as_posix().split("scripts/")[1].split(".py")[0], 
#     time.strftime("%Y-%m-%d"),
# )
# if not os.path.isdir(outdir):
#     os.makedirs(outdir)
# print("Output directory: {}".format(outdir))

# # Get timestamped output file prefix.
# timestamp = time.strftime("%y%m%d%H%M%S")
# prefix = "{}-{}".format(timestamp, script_name)
# print(prefix)

# def get_filename(filename):
#     """Add output directory path and timestamp prefix to filename."""
#     return os.path.join(outdir, "{}_{}".format(prefix, filename))


# # Save a timestamped copy of this file.
# shutil.copy(__file__, get_filename(".py"))

# # Save git hash
# git_hash = utils.git_revision_hash()
# git_url = "{}/commit/{}".format(utils.git_url(), git_hash)
# if git_hash and git_url and utils.is_git_clean():
#     print("Repository is clean.")
#     print("Code should be available at {}".format(git_url))
# else:
#     print("Unknown git revision.")
# info = open(get_filename("info.txt"), "w")
# info.write("git_hash: {}\n".format(git_hash))
# info.write("git_url: {}\n".format(git_url))
# info.close()

man = ScriptManager(datadir="/home/46h/sim_data/", path=pathlib.Path(__file__))
man.save_info()
man.save_script_copy()
print("Script info:")
pprint(man.get_info())


# Tracking
# --------------------------------------------------------------------------------------
madx_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 
    "data/SNS_RING_nux6.18_nuy6.18_dual_solenoid/LATTICE.lat",
)
madx_seq = "rnginjsol"
mass = mass_proton  # particle mass [GeV / c^2]
kin_energy = 0.800  # synchronous particle energy [GeV]


ring = SNS_RING()
ring.readMADX(madx_file, madx_seq)
ring.set_fringe_fields(False)
ring.initialize()

# Set solenoid field strengths. The total integrated field should be 0.6 [T*m].
solenoid_names = ["scbdsol_c13a", "scbdsol_c13b"]
for name in solenoid_names:
    node = ring.getNodeForName(name)
    B =  0.6 / (2.0 * node.getLength())
    node.setParam("B", B)
    print("{}: B={:.2f}, L={:.2f}".format(node.getName(), node.getParam("B"), node.getLength()))

# Analyze the one-turn transfer matrix.
bunch = Bunch()
bunch.mass(mass)
bunch.getSyncParticle().kinEnergy(kin_energy)
matrix_lattice = TEAPOT_MATRIX_Lattice_Coupled(ring, bunch)
tmat_params = matrix_lattice.getRingParametersDict()
eigtunes = tmat_params["eigtunes"]
eigvecs = tmat_params["eigvecs"]
print("eigtunes = {}".format(eigtunes))

# Add particles along each eigenvector.
v1 = eigvecs[:, 0]
v2 = eigvecs[:, 2]
psi1 = 0.0
psi2 = 0.0
J1 = 50.0e-6
J2 = 50.0e-6
x1 = np.sqrt(2.0 * J1) * np.real(v1 * np.exp(-1.0j * psi1))
x2 = np.sqrt(2.0 * J2) * np.real(v2 * np.exp(-1.0j * psi2))
print(x1)
print(x2)

bunch.deleteAllParticles()
for x in [x1, x2]:
    bunch.addParticle(x[0], x[1], x[2], x[3], 0.0, 0.0)
    
coords = []
for _ in range(2):
    coords.append([])

print("Tracking")
for turn in trange(100):
    ring.trackBunch(bunch)
    for i in range(len(coords)):
        coords[i].append([bunch.x(i), bunch.xp(i), bunch.y(i), bunch.yp(i)])
for i in range(len(coords)):
    coords[i] = np.array(coords[i])
    np.savetxt(man.get_filename("coords_mode{}.dat".format(i + 1)), coords[i]) 
    
    
# Print script info again for convenience.
print("Script info:")
pprint(man.get_info())