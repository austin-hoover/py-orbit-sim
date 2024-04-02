from pprint import pprint

import numpy as np

from bunch import Bunch
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice


# Create lattice
mad_filename = "LATTICE.lat"
mad_sequence = "rnginj"

lattice = TEAPOT_Lattice()
lattice.readMADX(mad_filename, mad_sequence)
lattice.initialize()

for node in lattice.getNodes():
    node.setUsageFringeFieldIN(False)
    node.setUsageFringeFieldOUT(False)

test_bunch = Bunch()
test_bunch.mass(0.938)
test_bunch.getSyncParticle().kinEnergy(1.3)
matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, test_bunch)
tmat_params = matrix_lattice.getRingParametersDict()

pprint(tmat_params)