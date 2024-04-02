"""This script estimates the phase rotation possible at full RF power.

We assume an initial uniform longitudinal charge density and small Gaussian 
energy spread. The energy spread is approximately what we could obtain
with a barrier RF cavity.

We increase the RF power to its maximum value. This is much larger than the 
typical SNS operating point.
"""
import math
import os
import time
from pprint import pprint

import numpy as np

from bunch import Bunch
from orbit.rf_cavities import RFLatticeModifications
from orbit.rf_cavities import RFNode
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice


# Setup
timestamp = time.strftime("%y%m%d%H%M%S")
output_dir = "./data_output/{}/".format(timestamp)
os.makedirs(output_dir)


# Create lattice
mad_filename = "./data_input/sns_ring_nux6.24_nuy6.15_mad.lattice"
mad_sequence = "RINGINJ"

lattice = TEAPOT_Lattice()
lattice.readMAD(mad_filename, mad_sequence)
lattice.initialize()

fringe_field_setting = True
for node in lattice.getNodes():
    node.setUsageFringeFieldIN(fringe_field_setting)
    node.setUsageFringeFieldOUT(fringe_field_setting)


# Add RF nodes.
z_to_phi = 2.0 * math.pi / lattice.getLength()
de_sync = 0.0

rf1a_hnum = 1.0
rf1b_hnum = 1.0
rf1c_hnum = 1.0
rf2_hnum  = 2.0

rf1a_voltage = +13.00e-06  # [GV]
rf1b_voltage = +13.00e-06  # [GV]
rf1c_voltage = +13.00e-06  # [GV]
rf2_voltage  = -0.00e-06  # [GV]

rf1a_phase = 0.0
rf1b_phase = 0.0
rf1c_phase = 0.0
rf2_phase  = 0.0
rf_length  = 0.0

rf_node_1a = RFNode.Harmonic_RFNode(z_to_phi, de_sync, rf1a_hnum, rf1a_voltage, rf1a_phase, rf_length, "RF1")
rf_node_1b = RFNode.Harmonic_RFNode(z_to_phi, de_sync, rf1b_hnum, rf1b_voltage, rf1b_phase, rf_length, "RF1")
rf_node_1c = RFNode.Harmonic_RFNode(z_to_phi, de_sync, rf1c_hnum, rf1c_voltage, rf1c_phase, rf_length, "RF1")
rf_node_2  = RFNode.Harmonic_RFNode(z_to_phi, de_sync, rf2_hnum,  rf2_voltage,  rf2_phase,  rf_length, "RF2")

RFLatticeModifications.addRFNode(lattice, 184.273, rf_node_1a)
RFLatticeModifications.addRFNode(lattice, 186.571, rf_node_1b)
RFLatticeModifications.addRFNode(lattice, 188.868, rf_node_1c)
RFLatticeModifications.addRFNode(lattice, 188.868, rf_node_2)


# Create line charge.
coords = np.zeros((5000, 6))

zmax = 0.5 * lattice.getLength()
coords[:, 4] = np.random.uniform(-zmax, zmax, size=coords.shape[0])
coords[:, 5] = np.random.normal(scale=0.003, size=coords.shape[0])

bunch = Bunch()
bunch.mass(0.938272)
bunch.getSyncParticle().kinEnergy(1.300)
for (x, xp, y, yp, z, dE) in coords:
    bunch.addParticle(x, xp, y, yp, z, dE)
    
# Track
n_turns = 1000
for turn in range(n_turns):
    print("turn={}".format(turn))
    
    filename = "bunch_{:04.0f}.dat".format(turn)
    filename = os.path.join(output_dir, filename)
    bunch.dumpBunch(filename)
    
    lattice.trackBunch(bunch)

print(timestamp)