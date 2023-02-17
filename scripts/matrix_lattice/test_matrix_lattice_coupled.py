"""Compute transfer matrix parameters."""
from __future__ import print_function
import os
from pprint import pprint 
import sys

import numpy as np

from bunch import Bunch 
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice_Coupled
from orbit.utils.consts import mass_proton


kin_energy = 1.0  # [GeV]
mass = mass_proton  # [GeV / c^2]
lattice_filename = '/home/46h/repo/accelerator-models/SNS/RING/SNS_ring_nux6.18_nuy6.18.lat'
lattice_seq = 'rnginj'

lattice = TEAPOT_Lattice()
lattice.readMADX(lattice_filename, lattice_seq)
for node in lattice.getNodes():
    node.setUsageFringeFieldIN(False)
    node.setUsageFringeFieldOUT(False)  
    if node.getName() == 'scbdsol':
        print('Found solenoid node:')
        print('    name = {}'.format(node.getName()))
        print('    type = {}'.format(node.getType()))
        print('    B = {} [1 / m]'.format(node.getParam('B')))
        
test_bunch = Bunch()
test_bunch.mass(mass)
test_bunch.getSyncParticle().kinEnergy(kin_energy)

print()
print('Courant-Snyder parameters (from MATRIX_Lattice:')
print('-----------------------------------------------')
matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, test_bunch)
params = matrix_lattice.getRingParametersDict()
pprint(params)

print()
print('Lebedev-Bogacz parameters (from MATRIX_Lattice_Coupled):')
print('--------------------------------------------------------`')
matrix_lattice = TEAPOT_MATRIX_Lattice_Coupled(lattice, test_bunch, parameterization='LB')
params = matrix_lattice.getRingParametersDict()
pprint(params)

print()
print('4x4 transfer matrix:')
M = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        M[i, j] = matrix_lattice.oneTurnMatrix.get(i, j)
print(M)


# Print painting path Re[v * exp(-i * mu)]
print()
print('Painting path:')
phase = np.radians(0.0)
v1, _, v2, _ = params['eigvecs'].T
for mode in [1, 2]:
    v = [v1, v2][mode - 1]
    print('mode = {}'.format(mode))
    print('phase = {}'.format(phase))
    print('Re[v * exp(-1j * phase)] = {}'.format(np.real(v * np.exp(-1.0j * phase))))
    print()