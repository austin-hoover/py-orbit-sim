from __future__ import print_function
import os
import sys
import time
import shutil
from pathlib import Path

import numpy as np

from bunch import Bunch
from orbit.bunch_generators import TwissContainer
from orbit.bunch_generators import WaterBagDist2D
from orbit.bunch_generators import WaterBagDist3D
from orbit.diagnostics import diagnostics
from orbit.lattice import AccActionsContainer
from orbit.py_linac.lattice import LinacAccLattice
from orbit.py_linac.lattice import LinacAccNodes

sys.path.append('/home/46h/py-orbit-sim/')
from pyorbit_sim import bunch_utils as bu
from pyorbit_sim.linac import Monitor
from pyorbit_sim.linac import track_bunch


# Setup
# ------------------------------------------------------------------------------
start = 0  # start node (name or index)
stop = 'MEBT:HZ04'  # stop node (name or index)

# Lattice
files = dict()
files['lattice_xml'] = '/home/46h/py-orbit-sim/pyorbit_sim/btf/data/xml/btf_lattice_default.xml'
files['lattice_coef'] = '/home/46h/py-orbit-sim/pyorbit_sim/btf/data/magnets/default_i2glCoefficients.csv'
files['lattice_mstate'] = '/home/46h/py-orbit-sim/pyorbit_sim/btf/data/mstate/TransmissionBS34_04212022.mstate'

# Bunch
files['input_bunch'] = '../input/realisticLEBT_50mA_42mA_8555k.dat'
bunch_freq = 402.5e6  # [Hz]
bunch_current = 0.042  # [A]

# Get timestamp and create output directory if it doesn't exixt.
outdir = './output/'
script_name = Path(__file__).stem
datestamp = time.strftime('%Y-%m-%d')
timestamp = time.strftime('%y%m%d%H%M%S')
prefix = '{}-{}'.format(timestamp, script_name)
if not os.path.isdir(outdir):
    os.makedirs(outdir)

    
# Sim
# ------------------------------------------------------------------------------
# Generate the BTF lattice
btf_lattice_generator = BTFLatticeGenerator(coef_filename=files['lattice_coef'])
btf_lattice_generator.init_lattice(
    xml=files['lattice_xml'], 
    beamlines=['MEBT1', 'MEBT2', 'MEBT3'], 
    max_drift_length=0.001,
)
btf_lattice_generator.update_quads_from_mstate(files['lattice_mstate'], value_type='current')
btf_lattice_generator.make_pmq_fields_overlap(z_step=0.001, verbose=True)
btf_lattice_generator.add_aperture_nodes(drift_step=0.1)
btf_lattice_generator.add_space_charge_nodes(
    grid_size_x=64, 
    grid_size_y=64,
    grid_size_z=64,
    path_length_min=0.005,
    n_bunches=3,
    freq=bunch_freq,
)
lattice = btf_lattice_generator.lattice

# Add diagnostics (dump bunch nodes)
diag_parent_nodes = [
    lattice.getNodeForName('MEBT:QH01'),
    lattice.getNodeForName('MEBT:QV02'),
    lattice.getNodeForName('MEBT:QH03'),
    lattice.getNodeForName('MEBT:QV04'),
]
for diag_parent_node in diag_parent_nodes:
    filename = './output/{}-bunch-{}.dat'.format(prefix, diag_parent_node.getName())
    diag_parent_node.addChildNode(
        diagnostics.DumpBunchNode(filename, verbose=True), 
        diag_parent_node.ENTRANCE, 
        part_index=0,
        place_in_part=AccActionsContainer.BEFORE,
    )
    
# Save node positions.
file = open('output/{}-node_positions.dat'.format(prefix), 'w')
file.write('node, position\n')
for node in lattice.getNodes():
    file.write('{}, {}\n'.format(node.getName(), node.getPosition()))
file.close()

# Write full lattice structure to file.
file = open(os.path.join(outdir, '{}-lattice_structure.txt'.format(prefix)), 'w')
file.write(lattice.structureToText())
file.close()
    
# Generate the bunch.
bunch = bu.load_bunch(
    filename=files['input_bunch'],
    file_format='pyorbit',
    verbose=True,
)
print('Charge = {}'.format(bunch.charge()))
print('Mass = {} [GeV / c^2]'.format(bunch.mass()))
print('Kinetic energy = {} [GeV]'.format(bunch.getSyncParticle().kinEnergy()))
print('Macrosize = {}'.format(bunch.macroSize()))


# Track the bunch.
monitor = Monitor(
    start_position=0.0,
    plotter=None,
    dispersion_flag=False,
    emit_norm_flag=False,
)
bunch.dumpBunch('./output/{}-bunch-START.dat'.format(prefix, diag_parent_node.getName()))
track_bunch(bunch, lattice, monitor=monitor, start=start, stop=stop, verbose=True)
bunch.dumpBunch('./output/{}-bunch-STOP.dat'.format(prefix, diag_parent_node.getName()))
monitor.write(filename='./output/{}-history.dat'.format(prefix), delimeter=',')