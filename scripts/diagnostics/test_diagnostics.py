"""Compute transfer matrix parameters."""
from __future__ import print_function
import os
import sys
import time

import numpy as np
from tqdm import trange

from bunch import Bunch 
from orbit.diagnostics import diagnostics
from orbit.diagnostics.diagnostics_lattice_modifications import add_diagnostics_node_as_child
from orbit.diagnostics.diagnostics_lattice_modifications import add_diagnostics_node
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.utils.consts import mass_proton

sys.path.append(os.getcwd())
from pyorbit_sim.bunch_utils import initialize_bunch
from pyorbit_sim.bunch_utils import set_bunch_coords


bunch, params_dict = initialize_bunch(mass=mass_proton, kin_energy=1.0)

X = np.random.normal(
    loc=[-0.005, 0.0, 0.0, 0.0, 0.0, 0.0],
    scale=[0.010, 0.010, 0.010, 0.010, 100.0, 0.0], 
    size=(10000, 6),
)
X[:, 5] = 0.0
bunch = set_bunch_coords(bunch, X)


lattice = TEAPOT_Lattice()
lattice_filename = '/home/46h/repo/accelerator-models/SNS/RING/SNS_ring_nux6.18_nuy6.18.lat'
lattice_seq = 'rnginj'
lattice.readMADX(lattice_filename, lattice_seq)
for node in lattice.getNodes():
    node.setUsageFringeFieldIN(False)
    node.setUsageFringeFieldOUT(False)


dnode = diagnostics.DanilovBunchCoordsNode(name='my_diag', skip=0, remember=5)
parent_node = lattice.getNodes()[0]
# add_diagnostics_node_as_child(dnode, parent_node=lattice.getNodes()[0])        
add_diagnostics_node(lattice, dnode, position=1.0)

for node in lattice.getNodes()[:10]:
    for child in node.getAllChildren():
        print(child.getName())

for _ in trange(10):
    lattice.trackBunch(bunch, params_dict)
    

dnode.package_data()
print(len(dnode.data))