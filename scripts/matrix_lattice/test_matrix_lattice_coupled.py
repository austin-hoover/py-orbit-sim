"""Compute transfer matrix parameters."""
from __future__ import print_function
import os
import pathlib
from pprint import pprint 
import sys

import numpy as np
import pandas as pd

from bunch import Bunch 
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_Ring
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice_Coupled
from orbit.utils.consts import mass_proton

sys.path.append(os.getcwd())
from pyorbit_sim.utils import ScriptManager


# Setup
man = ScriptManager(datadir="/home/46h/sim_data/", path=pathlib.Path(__file__))
man.save_info()
man.save_script_copy()
pprint(man.get_info())


mass = mass_proton  # particle mass [GeV / c^2]
kin_energy = 0.800  # synchronous particle energy [GeV]
madx_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 
    "SNS_RING_nux6.18_nuy6.18_dual_solenoid.lat",
)
madx_seq = "rnginjsol"

lattice = TEAPOT_Lattice()
lattice.readMADX(madx_file, madx_seq)
for node in lattice.getNodes():
    if node.getType() != "turn counter":
        node.setUsageFringeFieldIN(False)
        node.setUsageFringeFieldOUT(False)  
for name in ["scbdsol_c13a", "scbdsol_c13b"]:
    node = lattice.getNodeForName(name)
    B =  0.6 / (2.0 * node.getLength())
    # node.setParam("B", B)
    print("{}: B={:.2f}, L={:.2f}".format(node.getName(), node.getParam("B"), node.getLength()))
        
bunch = Bunch()
bunch.mass(mass)
bunch.getSyncParticle().kinEnergy(kin_energy)

print()
print("Courant-Snyder parameters (MATRIX_Lattice):")
print("-------------------------------------------")
matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, bunch)
params = matrix_lattice.getRingParametersDict()
pprint(params)

print()
print("Lebedev-Bogacz parameters (MATRIX_Lattice_Coupled):")
print("----------------------------------------------------`")
matrix_lattice = TEAPOT_MATRIX_Lattice_Coupled(lattice, bunch, parameterization="LB")
params = matrix_lattice.getRingParametersDict()
pprint(params)

print()
print("Tracked Twiss parameters (MATRIX_Lattice_Coupled):")
print("--------------------------------------------------")
data = matrix_lattice.getRingTwissData()
df = pd.DataFrame()
for key in data:
    df[key] = data[key]
print(df)

df.to_csv(man.get_filename("twiss.csv"))