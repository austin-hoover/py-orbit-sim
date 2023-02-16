from __future__ import print_function
import os
import sys
import time
from pathlib import Path
from pprint import pprint
import shutil

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import pandas as pd
from tqdm import tqdm
from tqdm import trange

from bunch import Bunch
from foil import Foil
from impedances import LImpedance
from impedances import TImpedance
from orbit.aperture import CircleApertureNode
from orbit.aperture import EllipseApertureNode
from orbit.aperture import RectangleApertureNode
from orbit.aperture import TeapotApertureNode
from orbit.bumps import TeapotBumpNode
from orbit.bumps import TeapotSimpleBumpNode
from orbit.bumps import BumpLatticeModifications
from orbit.bumps import bumps
from orbit.collimation import TeapotCollimatorNode
from orbit.collimation import addTeapotCollimatorNode
from orbit.envelope import DanilovEnvelope
from orbit.foils import TeapotFoilNode
from orbit.foils import addTeapotFoilNode
from orbit.impedances import addImpedanceNode
from orbit.impedances import LImpedance_Node
from orbit.impedances import FreqDep_LImpedance_Node
from orbit.impedances import BetFreqDep_LImpedance_Node
from orbit.impedances import TImpedance_Node
from orbit.impedances import FreqDep_TImpedance_Node
from orbit.impedances import BetFreqDep_TImpedance_Node
from orbit.injection import addTeapotInjectionNode
from orbit.injection import InjectParts
from orbit.injection import TeapotInjectionNode
from orbit.injection.distributions import JohoTransverse
from orbit.injection.distributions import JohoLongitudinal
from orbit.injection.distributions import SNSESpreadDist
from orbit.injection.distributions import UniformLongDist
from orbit.lattice import AccNode
from orbit.rf_cavities import RFNode
from orbit.rf_cavities import RFLatticeModifications
from orbit.space_charge.envelope import set_env_solver_nodes
from orbit.space_charge.sc1d import addLongitudinalSpaceChargeNode
from orbit.space_charge.sc1d import SC1D_AccNode
from orbit.space_charge import sc2p5d
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.space_charge import sc2dslicebyslice
from orbit.space_charge.sc2dslicebyslice.scLatticeModifications import setSC2DSliceBySliceAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice_Coupled
from orbit.teapot import TEAPOT_Ring
from orbit.time_dep import TIME_DEP_Lattice
from orbit.time_dep.waveform import ConstantWaveform
from orbit.time_dep.waveform import SquareRootWaveform
from orbit.utils import consts
import orbit_mpi
from spacecharge import LSpaceChargeCalc
from spacecharge import Boundary2D
from spacecharge import SpaceChargeCalc2p5D
from spacecharge import SpaceChargeCalcSliceBySlice2D

## Local
import injection_control as inj

sys.path.append('/home/46h/repo/py-orbit-sim/')
from pyorbit_sim.lattice import get_matrix_lattice
from pyorbit_sim.lattice import get_matrix_lattice_dispersion
from pyorbit_sim.lattice import get_matrix_lattice_twiss
from pyorbit_sim.lattice import set_node_fringe
from pyorbit_sim.lattice import split_node



# Setup
# ------------------------------------------------------------------------------

# MPI
_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)


# Output file directory and prefix (timestamp)

outdir = "./_output/"
script_name = Path(__file__).stem
timestamp = ''
if _mpi_rank == 0:
    timestamp = time.strftime("%y%m%d%H%M%S")
timestamp = orbit_mpi.MPI_Bcast(timestamp, orbit_mpi.mpi_datatype.MPI_CHAR, 0, _mpi_comm)
prefix = "{}-{}".format(timestamp, script_name)
if _mpi_rank == 0:
    print(prefix)
    
def get_filename(filename):
    """Add output directory path and timestamp prefix to filename."""
    return os.path.join(outdir, '{}_{}'.format(prefix, filename))

## Save a timestamped copy of this file. (It would be better to use version-control.)
shutil.copy(__file__, get_filename('{}.py'.format(script_name)))

    
# Lattice parameters
madx_file = "_input/SNS_RING_nux6.18_nuy6.18.lat"
madx_seq = "rnginj"
X_FOIL = 0.0486  # horizontal position of foil [m]
Y_FOIL = 0.0460  # vertical position of foil [m]
    
# Bunch/simulation parameters
n_inj_turns = 501  # number of turns to inject
n_stored_turns = 0  # number of turns to store the beam after injection
kin_energy = 0.800  # synchronous particle energy [GeV]
mass = consts.mass_proton  # particle mass [GeV / c^2]
bunch_length_frac = (43.0 / 64.0)  # macropulse length relative to ring length 
n_macros_total = int(1.0e5)  # final number of macroparticles
n_macros_per_turn = int(n_macros_total / n_inj_turns)  # macroparticles per minipulse
nominal_n_inj_turns = 1000.0
nominal_intensity = 1.5e14

## Initial coordinates of closed orbit at injection point [x, x', y, y'].
inj_coords_t0 = np.zeros(4)
inj_coords_t0[0] = X_FOIL
inj_coords_t0[1] = 0.0
inj_coords_t0[2] = Y_FOIL
inj_coords_t0[3] = 0.0

## Final coordinates of closed orbit at injection point  [x, x', y, y'].
inj_coords_t1 = np.zeros(4)
inj_coords_t1[0] = X_FOIL - 0.035
inj_coords_t1[1] = 0.0
inj_coords_t1[2] = Y_FOIL
inj_coords_t1[3] = -0.002

## Which turns to save the phase space coordinates.
skip_inj = 1



# Lattice setup and linear analysis
# ------------------------------------------------------------------------------

# Create time-dependent lattice.
ring = TIME_DEP_Lattice() 
ring.readMADX(madx_file, madx_seq)
ring.initialize()
ring_length = ring.getLength()

# (Note that TEAPOT_Ring has been changed: children are added on instantiation.
# This means that the nodes cannot be split. This is not a problem for
# TEAPOT_Lattice.)


# Linear lattice analysis (uncoupled)
    
## Turn off fringe fields. (This is affecting the MATRIX_Lattice tune calculation
## below. But shouldn't MATRIX_Lattice just extract the linear transfer matrix
# from each node? TODO: Figure out what's going on.)
for node in ring.getNodes():
    set_node_fringe(node, False, verbose=False)
    
## Construct the MATRIX_Lattice.
test_bunch = Bunch()
test_bunch.mass(mass)
test_bunch.getSyncParticle().kinEnergy(kin_energy)
matrix_lattice = TEAPOT_MATRIX_Lattice(ring, test_bunch)

## Save the Twiss parameters throughout the lattice.
twiss = get_matrix_lattice_twiss(matrix_lattice)
dispersion = get_matrix_lattice_dispersion(matrix_lattice)
if _mpi_rank == 0:
    filename = get_filename('lattice_twiss.dat')
    twiss.to_csv(filename, sep=' ')
    # print('Wrote lattice twiss to {}'.format(filename))
    
    filename = get_filename('lattice_dispersion.dat')
    dispersion.to_csv(filename, sep=' ')
    # print('Wrote lattice dispersion to {}'.format(filename))
    
## Analyze the transfer matrix.
tmat_params = matrix_lattice.getRingParametersDict()
# if _mpi_rank == 0:
    # print('Transfer matrix parameters (uncoupled):')
    # pprint(tmat_params)

    
# Linear lattice analysis (coupled)

## Construct the MATRIX_Lattice.
matrix_lattice = TEAPOT_MATRIX_Lattice_Coupled(ring, test_bunch, parameterization='LB')

## Analyze the transfer matrix.
tmat_params = matrix_lattice.getRingParametersDict()
# if _mpi_rank == 0:
#     print('Transfer matrix parameters (coupled):')
#     pprint(tmat_params)


# Turn off fringe fields. (See comment above.)
for node in ring.getNodes():
    set_node_fringe(node, True, verbose=False)

# Save injection region node positions.
if _mpi_rank == 0:
    file = open(get_filename('inj_region_node_positions.txt'), 'w')
    for node in ring.getNodes():
        start, stop = ring.getNodePositionsDict()[node]
        file.write("{} {} {}\n".format(node.getName(), start, stop))
    file.close()


    
# Bunch setup
# ------------------------------------------------------------------------------
    
# Initialize bunch
bunch = Bunch()
bunch.mass(mass)
sync_part = bunch.getSyncParticle()
sync_part.kinEnergy(kin_energy)
lostbunch = Bunch()
lostbunch.addPartAttr('LostParticleAttributes')
params_dict = {'bunch': bunch, 'lostbunch': lostbunch}

# Compute macrosize.
nominal_minipulse_intensity = nominal_intensity / nominal_n_inj_turns
nominal_bunch_length_frac = (50.0 / 64.0)
bunch_length_factor = bunch_length_frac / nominal_bunch_length_frac
minipulse_intensity = nominal_minipulse_intensity * bunch_length_factor
intensity = minipulse_intensity * n_inj_turns
macro_size = intensity / n_inj_turns / n_macros_per_turn
bunch.macroSize(macro_size)


# Injected phase space distribution

inj_dist = dict()

## Transverse injected distribution (horizontal)
inj_dist['x'] = {
    'centroid': [X_FOIL, 0.0],  # (x, xp)
    'alpha': -0.924,
    'beta': 3.71,  # [m / rad]
    'order': 9.0,
    'eps_rms': 0.467e-6  # [m * rad] (TODO: check this)
}
inj_dist['x']['eps_lim'] = inj_dist['x']['eps_rms'] * 2.0 * (inj_dist['x']['order'] + 1.0)
inj_dist['x']['gen'] = JohoTransverse(
    order=inj_dist['x']['order'],
    alpha=inj_dist['x']['alpha'],
    beta=inj_dist['x']['beta'],
    emitlim=inj_dist['x']['eps_lim'],
    centerpos=inj_dist['x']['centroid'][0],
    centermom=inj_dist['x']['centroid'][1],
)

## Transverse injected distribution (vertical)
inj_dist['y'] = {
    'centroid': [Y_FOIL, 0.0],  # (y, yp)
    'alpha': -0.5,
    'beta': 4.86,  # [m/rad]
    'order': 9.0,
    'eps_rms': 0.300e-6,  # [m * rad] (TODO: check this)
}
inj_dist['y']['eps_lim'] = inj_dist['y']['eps_rms'] * 2.0 * (inj_dist['y']['order'] + 1.0)
inj_dist['y']['gen'] = JohoTransverse(
    order=inj_dist['y']['order'],
    alpha=inj_dist['y']['alpha'],
    beta=inj_dist['y']['beta'],
    emitlim=inj_dist['y']['eps_lim'],
    centerpos=inj_dist['y']['centroid'][0],
    centermom=inj_dist['y']['centroid'][1],
)

## Longitudinal injected distribution
seconds_per_turn = ring_length / (sync_part.beta() * consts.speed_of_light)
inj_dist['z'] = {
    'zmin': -0.5 * bunch_length_frac * ring_length,
    'zmax': +0.5 * bunch_length_frac * ring_length,
    'tailfraction': 0.0,
    'eparams': {
        'mean': sync_part.kinEnergy(),
        'sigma': 0.0005,
        'trunc': 1.0,
        'min': sync_part.kinEnergy() - 0.0025,
        'max': sync_part.kinEnergy() + 0.0025,
    },
    'ecparams': {
        'mean': 0.0,
        'sigma': 0.000000001,
        'trunc': 1.0,
        'min': -0.0035,
        'max': +0.0035,
        'drifti': 0.0,
        'driftf': 0.0,
        'drifttime': 1000.0 * n_inj_turns * seconds_per_turn  # [ms]
    },
    'esparams': {
        'nu': 100.0,
        'phase': 0.0,
        'max': 0.0,
        'nulltime': 0.0,
    },
}
inj_dist['z']['gen'] = SNSESpreadDist(
    lattlength=ring_length,
    zmin=inj_dist['z']['zmin'],
    zmax=inj_dist['z']['zmax'],
    tailfraction=inj_dist['z']['tailfraction'],
    sp=sync_part,
    emean=inj_dist['z']['eparams']['mean'],
    esigma=inj_dist['z']['eparams']['sigma'],
    etrunc=inj_dist['z']['eparams']['trunc'],
    emin=inj_dist['z']['eparams']['min'],
    emax=inj_dist['z']['eparams']['max'],
    ecparams=inj_dist['z']['ecparams'],
    esparams=inj_dist['z']['esparams'],
)
# if _mpi_rank == 0:
#     print('Injected (linac) distribution parameters:')
#     pprint(inj_dist)


# # Apertures and displacements - injection chicane
# # ------------------------------------------------------------------------------
# xcenter = 0.100
# xb10m = 0.0
# xb11m = 0.09677
# xb12m = 0.08899
# xb13m = 0.08484
# ycenter = 0.023
# bumpwave = ConstantWaveform(amplitude=1.0)

# xb10i = 0.0
# apb10xi = -xb10i
# apb10yi = ycenter
# rb10i = 0.1095375
# appb10i = CircleApertureNode(rb10i, 244.812, apb10xi, apb10yi, name="b10i")

# xb10f = 0.022683
# apb10xf = -xb10f
# apb10yf = ycenter
# rb10f = 0.1095375
# appb10f = CircleApertureNode(rb10f, 245.893, apb10xf, apb10yf, name="b10f")

# mag10x = (xb10i + xb10f) / 2.0 - xb10m
# cdb10i = TeapotSimpleBumpNode(bunch, mag10x, 0.0, -ycenter, 0.0, bumpwave, "mag10bumpi")
# cdb10f = TeapotSimpleBumpNode(bunch, -mag10x, 0.0, ycenter, 0.0, bumpwave, "mag10bumpf")

# xb11i = 0.074468
# apb11xi = xcenter - xb11i
# apb11yi = ycenter
# rb11i = 0.1095375
# appb11i = CircleApertureNode(rb11i, 247.265, apb11xi, apb11yi, name="b11i")

# xfoil = 0.099655
# apfoilx = xcenter - xfoil
# apfoily = ycenter
# rfoil = 0.1095375
# appfoil = CircleApertureNode(rfoil, 248.009, apfoilx, apfoily, name="bfoil")

# mag11ax = (xb11i + xfoil) / 2.0 - xb11m
# cdb11i = TeapotSimpleBumpNode(bunch, mag11ax, 0.0, -ycenter, 0.0, bumpwave, "mag11bumpi")
# cdfoila = TeapotSimpleBumpNode(bunch, -mag11ax, 0.0, ycenter, 0.0, bumpwave, "foilbumpa")

# xb11f = 0.098699
# apb11xf = xcenter - xb11f
# apb11yf = ycenter
# rb11f = 0.1095375
# appb11f = CircleApertureNode(rb11f, 0.195877, apb11xf, apb11yf, name="b11f")

# mag11bx = (xfoil + xb11f) / 2.0 - xb11m
# cdfoilb = TeapotSimpleBumpNode(bunch, mag11bx, 0.0, -ycenter, 0.0, bumpwave, "foilbumpb")
# cdb11f = TeapotSimpleBumpNode(bunch, -mag11bx, 0.0, ycenter, 0.0, bumpwave, "mag11bumpf")

# xb12i = 0.093551
# apb12xi = xcenter - xb12i
# apb12yi = ycenter
# rb12i = 0.1095375
# appb12i = CircleApertureNode(rb12i, 1.08593, apb12xi, apb12yi, name="b12i")

# xb12f = 0.05318
# apb12xf = xcenter - xb12f
# apb12yf = ycenter
# rb12f = 0.1174750
# appb12f = CircleApertureNode(rb12f, 1.99425, apb12xf, apb12yf, name="b12f")

# mag12x = (xb12i + xb12f) / 2.0 - xb12m
# cdb12i = TeapotSimpleBumpNode(bunch, mag12x, 0.0, -ycenter, 0.0, bumpwave, "mag12bumpi")
# cdb12f = TeapotSimpleBumpNode(bunch, -mag12x, 0.0, ycenter, 0.0, bumpwave, "mag12bumpf")

# xb13i = 0.020774
# apb13xi = xcenter - xb13i
# apb13yi = ycenter
# h13xi = 0.1913
# v13xi = 0.1016
# appb13i = RectangleApertureNode(h13xi, v13xi, 3.11512, apb13xi, apb13yi, name="b13i")

# xb13f = 0.0
# apb13xf = xcenter - xb13f
# apb13yf = ycenter
# h13xf = 0.1913
# v13xf = 0.1016
# appb13f = RectangleApertureNode(h13xf, v13xf, 4.02536, apb13xf, apb13yf, name="b13f")

# mag13x = (xb13i + xb13f) / 2.0 - xb13m
# cdb13i = TeapotSimpleBumpNode(bunch, mag13x, 0.0, -ycenter, 0.0, bumpwave, "mag13bumpi")
# cdb13f = TeapotSimpleBumpNode(bunch, -mag13x, 0.0, ycenter, 0.0, bumpwave, "mag13bumpf")

# dha10 = ring.getNodeForName("dh_a10")
# dha11a = ring.getNodeForName("dh_a11a")
# dha11b = ring.getNodeForName("dh_a11b")
# dha12 = ring.getNodeForName("dh_a12")
# dha13 = ring.getNodeForName("dh_a13")

# dha10.addChildNode(appb10i, AccNode.ENTRANCE)
# dha10.addChildNode(cdb10i, AccNode.ENTRANCE)
# dha10.addChildNode(cdb10f, AccNode.EXIT)
# dha10.addChildNode(appb10f, AccNode.EXIT)
# dha11a.addChildNode(appb11i, AccNode.ENTRANCE)
# dha11a.addChildNode(cdb11i, AccNode.ENTRANCE)
# dha11a.addChildNode(cdfoila, AccNode.EXIT)
# dha11a.addChildNode(appfoil, AccNode.EXIT)
# dha11b.addChildNode(cdfoilb, AccNode.ENTRANCE)
# dha11b.addChildNode(cdb11f, AccNode.EXIT)
# dha11b.addChildNode(appb11f, AccNode.EXIT)
# dha12.addChildNode(appb12i, AccNode.ENTRANCE)
# dha12.addChildNode(cdb12i, AccNode.ENTRANCE)
# dha12.addChildNode(cdb12f, AccNode.EXIT)
# dha12.addChildNode(appb12f, AccNode.EXIT)
# dha13.addChildNode(appb13i, AccNode.ENTRANCE)
# dha13.addChildNode(cdb13i, AccNode.ENTRANCE)
# dha13.addChildNode(cdb13f, AccNode.EXIT)
# dha13.addChildNode(appb13f, AccNode.EXIT)


# # Apertures and collimators - around the ring
# #------------------------------------------------------------------------------
# a115p8 = 0.1158
# b078p7 = 0.0787
# a080p0 = 0.0800
# b048p0 = 0.0480

# r062 = 0.0625
# r100 = 0.1000
# r120 = 0.1200
# r125 = 0.1250
# r130 = 0.1300
# r140 = 0.1400

# app06200 = CircleApertureNode(r062,  58.21790, 0.0, 0.0, name = "s1")
# app06201 = CircleApertureNode(r062,  65.87790, 0.0, 0.0, name = "s2")

# app10000 = CircleApertureNode(r100,  10.54740, 0.0, 0.0, name = "bp100")
# app10001 = CircleApertureNode(r100,  10.97540, 0.0, 0.0, name = "bp100")
# app10002 = CircleApertureNode(r100,  13.57190, 0.0, 0.0, name = "bp100")
# app10003 = CircleApertureNode(r100,  15.39140, 0.0, 0.0, name = "21cmquad")
# app10004 = CircleApertureNode(r100,  19.39150, 0.0, 0.0, name = "21cmquad")
# app10005 = CircleApertureNode(r100,  23.39170, 0.0, 0.0, name = "21cmquad")
# app10006 = CircleApertureNode(r100,  31.39210, 0.0, 0.0, name = "21cmquad")
# app10007 = CircleApertureNode(r100,  39.39250, 0.0, 0.0, name = "21cmquad")
# app10008 = CircleApertureNode(r100,  43.39270, 0.0, 0.0, name = "21cmquad")
# app10009 = CircleApertureNode(r100,  47.39290, 0.0, 0.0, name = "21cmquad")
# app10010 = CircleApertureNode(r100,  48.86630, 0.0, 0.0, name = "bp100")
# app10011 = CircleApertureNode(r100,  50.37710, 0.0, 0.0, name = "P1a")
# app10012 = CircleApertureNode(r100,  51.19660, 0.0, 0.0, name = "scr1c")
# app10013 = CircleApertureNode(r100,  51.24100, 0.0, 0.0, name = "scr2c")
# app10014 = CircleApertureNode(r100,  51.39470, 0.0, 0.0, name = "scr3c")
# app10015 = CircleApertureNode(r100,  51.43910, 0.0, 0.0, name = "scr4c")
# app10016 = CircleApertureNode(r100,  51.62280, 0.0, 0.0, name = "p2shield")
# app10017 = CircleApertureNode(r100,  53.74280, 0.0, 0.0, name = "p2shield")
# app10018 = CircleApertureNode(r100,  54.75640, 0.0, 0.0, name = "bp100")
# app10019 = CircleApertureNode(r100,  71.16320, 0.0, 0.0, name = "bp100")
# app10020 = CircleApertureNode(r100,  73.35170, 0.0, 0.0, name = "bp100")
# app10021 = CircleApertureNode(r100,  75.60170, 0.0, 0.0, name = "bp100")
# app10022 = CircleApertureNode(r100,  76.79720, 0.0, 0.0, name = "bp100")
# app10023 = CircleApertureNode(r100,  77.39290, 0.0, 0.0, name = "21cmquad")
# app10024 = CircleApertureNode(r100,  81.39310, 0.0, 0.0, name = "21cmquad")
# app10025 = CircleApertureNode(r100,  85.39330, 0.0, 0.0, name = "21cmquad")
# app10026 = CircleApertureNode(r100,  93.39370, 0.0, 0.0, name = "21cmquad")
# app10027 = CircleApertureNode(r100, 101.39400, 0.0, 0.0, name = "21cmquad")
# app10028 = CircleApertureNode(r100, 105.39400, 0.0, 0.0, name = "21cmquad")
# app10029 = CircleApertureNode(r100, 109.39400, 0.0, 0.0, name = "21cmquad")
# app10030 = CircleApertureNode(r100, 110.49000, 0.0, 0.0, name = "bp100")
# app10031 = CircleApertureNode(r100, 112.69100, 0.0, 0.0, name = "bp100")
# app10032 = CircleApertureNode(r100, 114.82200, 0.0, 0.0, name = "bp100")
# app10033 = CircleApertureNode(r100, 118.38600, 0.0, 0.0, name = "bp100")
# app10034 = CircleApertureNode(r100, 120.37900, 0.0, 0.0, name = "bp100")
# app10035 = CircleApertureNode(r100, 122.21700, 0.0, 0.0, name = "bp100")
# app10036 = CircleApertureNode(r100, 124.64400, 0.0, 0.0, name = "bp100")
# app10037 = CircleApertureNode(r100, 127.77400, 0.0, 0.0, name = "bp100")
# app10038 = CircleApertureNode(r100, 132.53100, 0.0, 0.0, name = "bp100")
# app10039 = CircleApertureNode(r100, 136.10400, 0.0, 0.0, name = "bp100")
# app10040 = CircleApertureNode(r100, 138.79900, 0.0, 0.0, name = "bp100")
# app10041 = CircleApertureNode(r100, 139.39400, 0.0, 0.0, name = "21cmquad")
# app10042 = CircleApertureNode(r100, 143.39500, 0.0, 0.0, name = "21cmquad")
# app10043 = CircleApertureNode(r100, 147.39500, 0.0, 0.0, name = "21cmquad")
# app10044 = CircleApertureNode(r100, 155.39500, 0.0, 0.0, name = "21cmquad")
# app10045 = CircleApertureNode(r100, 163.39600, 0.0, 0.0, name = "21cmquad")
# app10046 = CircleApertureNode(r100, 167.39600, 0.0, 0.0, name = "21cmquad")
# app10047 = CircleApertureNode(r100, 171.39600, 0.0, 0.0, name = "21cmquad")
# app10048 = CircleApertureNode(r100, 173.70900, 0.0, 0.0, name = "bp100")
# app10049 = CircleApertureNode(r100, 175.93900, 0.0, 0.0, name = "bp100")
# app10050 = CircleApertureNode(r100, 180.38700, 0.0, 0.0, name = "bp100")
# app10051 = CircleApertureNode(r100, 182.12700, 0.0, 0.0, name = "bp100")
# app10052 = CircleApertureNode(r100, 184.27300, 0.0, 0.0, name = "bp100")
# app10053 = CircleApertureNode(r100, 186.57100, 0.0, 0.0, name = "bp100")
# app10054 = CircleApertureNode(r100, 188.86800, 0.0, 0.0, name = "bp100")
# app10055 = CircleApertureNode(r100, 191.16500, 0.0, 0.0, name = "bp100")
# app10056 = CircleApertureNode(r100, 194.53200, 0.0, 0.0, name = "bp100")
# app10057 = CircleApertureNode(r100, 196.61400, 0.0, 0.0, name = "bp100")
# app10058 = CircleApertureNode(r100, 199.47500, 0.0, 0.0, name = "bp100")
# app10059 = CircleApertureNode(r100, 201.39600, 0.0, 0.0, name = "21cmquad")
# app10060 = CircleApertureNode(r100, 205.39600, 0.0, 0.0, name = "21cmquad")
# app10061 = CircleApertureNode(r100, 209.39600, 0.0, 0.0, name = "21cmquad")
# app10062 = CircleApertureNode(r100, 217.39700, 0.0, 0.0, name = "21cmquad")
# app10063 = CircleApertureNode(r100, 225.39700, 0.0, 0.0, name = "21cmquad")
# app10064 = CircleApertureNode(r100, 229.39700, 0.0, 0.0, name = "21cmquad")
# app10065 = CircleApertureNode(r100, 233.39700, 0.0, 0.0, name = "21cmquad")
# app10066 = CircleApertureNode(r100, 234.87800, 0.0, 0.0, name = "bp100")
# app10067 = CircleApertureNode(r100, 236.87700, 0.0, 0.0, name = "bp100")
# app10068 = CircleApertureNode(r100, 238.74100, 0.0, 0.0, name = "bp100")
# app10069 = CircleApertureNode(r100, 242.38900, 0.0, 0.0, name = "bp100")

# app12000 = CircleApertureNode(r120,   6.89986, 0.0, 0.0, name = "bp120")
# app12001 = CircleApertureNode(r120,   8.52786, 0.0, 0.0, name = "bp120")
# app12002 = CircleApertureNode(r120,  57.20790, 0.0, 0.0, name = "s1shield")
# app12003 = CircleApertureNode(r120,  59.40790, 0.0, 0.0, name = "s1shield")
# app12004 = CircleApertureNode(r120,  64.86790, 0.0, 0.0, name = "s2shield")
# app12005 = CircleApertureNode(r120,  67.06790, 0.0, 0.0, name = "s2shield")
# app12006 = CircleApertureNode(r120, 116.75800, 0.0, 0.0, name = "bp120")
# app12007 = CircleApertureNode(r120, 130.90300, 0.0, 0.0, name = "bp120")
# app12008 = CircleApertureNode(r120, 178.75900, 0.0, 0.0, name = "bp120")
# app12009 = CircleApertureNode(r120, 192.90400, 0.0, 0.0, name = "bp120")
# app12010 = CircleApertureNode(r120, 240.76100, 0.0, 0.0, name = "bp120")

# app12500 = CircleApertureNode(r125,  27.37140, 0.0, 0.0, name = "26cmquad")
# app12501 = CircleApertureNode(r125,  35.37180, 0.0, 0.0, name = "26cmquad")
# app12502 = CircleApertureNode(r125,  89.37300, 0.0, 0.0, name = "26cmquad")
# app12503 = CircleApertureNode(r125,  97.37330, 0.0, 0.0, name = "26cmquad")
# app12504 = CircleApertureNode(r125, 151.37400, 0.0, 0.0, name = "26cmquad")
# app12505 = CircleApertureNode(r125, 159.37500, 0.0, 0.0, name = "26cmquad")
# app12506 = CircleApertureNode(r125, 213.37600, 0.0, 0.0, name = "26cmquad")
# app12507 = CircleApertureNode(r125, 221.37600, 0.0, 0.0, name = "26cmquad")

# app13000 = CircleApertureNode(r130,  60.41790, 0.0, 0.0, name = "bp130")
# app13001 = CircleApertureNode(r130,  64.42290, 0.0, 0.0, name = "bp130")
# app13002 = CircleApertureNode(r130,  68.07790, 0.0, 0.0, name = "bp130")
# app13003 = CircleApertureNode(r130,  68.90140, 0.0, 0.0, name = "bp130")
# app13004 = CircleApertureNode(r130,  70.52940, 0.0, 0.0, name = "bp130")

# app14000 = CircleApertureNode(r140,   7.43286, 0.0, 0.0, name = "30cmquad")
# app14001 = CircleApertureNode(r140,   7.85486, 0.0, 0.0, name = "30cmquad")
# app14002 = CircleApertureNode(r140,  55.42940, 0.0, 0.0, name = "30cmquad")
# app14003 = CircleApertureNode(r140,  55.85140, 0.0, 0.0, name = "30cmquad")
# app14004 = CircleApertureNode(r140,  56.38440, 0.0, 0.0, name = "30cmquad")
# app14005 = CircleApertureNode(r140,  60.86290, 0.0, 0.0, name = "bp140")
# app14006 = CircleApertureNode(r140,  62.64290, 0.0, 0.0, name = "bp140")
# app14007 = CircleApertureNode(r140,  63.97790, 0.0, 0.0, name = "bp140")
# app14008 = CircleApertureNode(r140,  69.43440, 0.0, 0.0, name = "30cmquad")
# app14009 = CircleApertureNode(r140,  69.85640, 0.0, 0.0, name = "30cmquad")
# app14010 = CircleApertureNode(r140, 117.43100, 0.0, 0.0, name = "30cmquad")
# app14011 = CircleApertureNode(r140, 117.85300, 0.0, 0.0, name = "30cmquad")
# app14012 = CircleApertureNode(r140, 131.43600, 0.0, 0.0, name = "30cmquad")
# app14013 = CircleApertureNode(r140, 131.85800, 0.0, 0.0, name = "30cmquad")
# app14014 = CircleApertureNode(r140, 179.43200, 0.0, 0.0, name = "30cmquad")
# app14015 = CircleApertureNode(r140, 179.85400, 0.0, 0.0, name = "30cmquad")
# app14016 = CircleApertureNode(r140, 193.43700, 0.0, 0.0, name = "30cmquad")
# app14017 = CircleApertureNode(r140, 193.85900, 0.0, 0.0, name = "30cmquad")
# app14018 = CircleApertureNode(r140, 241.43400, 0.0, 0.0, name = "30cmquad")
# app14019 = CircleApertureNode(r140, 241.85600, 0.0, 0.0, name = "30cmquad")

# appell00 = EllipseApertureNode(a115p8, b078p7,  16.9211, 0.0, 0.0, name = "arcdipole")
# appell01 = EllipseApertureNode(a115p8, b078p7,  20.9213, 0.0, 0.0, name = "arcdipole")
# appell02 = EllipseApertureNode(a115p8, b078p7,  24.9215, 0.0, 0.0, name = "arcdipole")
# appell03 = EllipseApertureNode(a115p8, b078p7,  28.9217, 0.0, 0.0, name = "arcdipole")
# appell04 = EllipseApertureNode(a115p8, b078p7,  32.9219, 0.0, 0.0, name = "arcdipole")
# appell05 = EllipseApertureNode(a115p8, b078p7,  36.9221, 0.0, 0.0, name = "arcdipole")
# appell06 = EllipseApertureNode(a115p8, b078p7,  40.9222, 0.0, 0.0, name = "arcdipole")
# appell07 = EllipseApertureNode(a115p8, b078p7,  44.9224, 0.0, 0.0, name = "arcdipole")
# appell08 = EllipseApertureNode(a080p0, b048p0,  51.9428, 0.0, 0.0, name = "p2")
# appell09 = EllipseApertureNode(a115p8, b078p7,  78.9226, 0.0, 0.0, name = "arcdipole")
# appell10 = EllipseApertureNode(a115p8, b078p7,  82.9228, 0.0, 0.0, name = "arcdipole")
# appell11 = EllipseApertureNode(a115p8, b078p7,  86.9230, 0.0, 0.0, name = "arcdipole")
# appell12 = EllipseApertureNode(a115p8, b078p7,  90.9232, 0.0, 0.0, name = "arcdipole")
# appell13 = EllipseApertureNode(a115p8, b078p7,  94.9234, 0.0, 0.0, name = "arcdipole")
# appell14 = EllipseApertureNode(a115p8, b078p7,  98.9236, 0.0, 0.0, name = "arcdipole")
# appell15 = EllipseApertureNode(a115p8, b078p7, 102.9240, 0.0, 0.0, name = "arcdipole")
# appell16 = EllipseApertureNode(a115p8, b078p7, 106.9240, 0.0, 0.0, name = "arcdipole")
# appell17 = EllipseApertureNode(a115p8, b078p7, 140.9240, 0.0, 0.0, name = "arcdipole")
# appell18 = EllipseApertureNode(a115p8, b078p7, 144.9240, 0.0, 0.0, name = "arcdipole")
# appell19 = EllipseApertureNode(a115p8, b078p7, 148.9250, 0.0, 0.0, name = "arcdipole")
# appell20 = EllipseApertureNode(a115p8, b078p7, 152.9250, 0.0, 0.0, name = "arcdipole")
# appell21 = EllipseApertureNode(a115p8, b078p7, 156.9250, 0.0, 0.0, name = "arcdipole")
# appell22 = EllipseApertureNode(a115p8, b078p7, 160.9250, 0.0, 0.0, name = "arcdipole")
# appell23 = EllipseApertureNode(a115p8, b078p7, 164.9250, 0.0, 0.0, name = "arcdipole")
# appell24 = EllipseApertureNode(a115p8, b078p7, 168.9260, 0.0, 0.0, name = "arcdipole")
# appell25 = EllipseApertureNode(a115p8, b078p7, 202.9260, 0.0, 0.0, name = "arcdipole")
# appell26 = EllipseApertureNode(a115p8, b078p7, 206.9260, 0.0, 0.0, name = "arcdipole")
# appell27 = EllipseApertureNode(a115p8, b078p7, 210.9260, 0.0, 0.0, name = "arcdipole")
# appell28 = EllipseApertureNode(a115p8, b078p7, 214.9260, 0.0, 0.0, name = "arcdipole")
# appell29 = EllipseApertureNode(a115p8, b078p7, 218.9260, 0.0, 0.0, name = "arcdipole")
# appell30 = EllipseApertureNode(a115p8, b078p7, 222.9270, 0.0, 0.0, name = "arcdipole")
# appell31 = EllipseApertureNode(a115p8, b078p7, 226.9270, 0.0, 0.0, name = "arcdipole")
# appell32 = EllipseApertureNode(a115p8, b078p7, 230.9270, 0.0, 0.0, name = "arcdipole")

# p1    = TeapotCollimatorNode(0.60000, 3, 1.00, 2, 0.100, 0.100, 0.0, 0.0,   0.0, 50.3771)
# scr1t = TeapotCollimatorNode(0.00450, 5, 1.00, 3, 0.100, 0.000, 0.0, 0.0,   0.0, 51.1921)
# scr1c = TeapotCollimatorNode(0.01455, 4, 1.00, 3, 0.100, 0.000, 0.0, 0.0,   0.0, 51.1966)
# scr2t = TeapotCollimatorNode(0.00450, 5, 1.00, 3, 0.100, 0.000, 0.0, 0.0,  90.0, 51.2365)
# scr2c = TeapotCollimatorNode(0.01455, 4, 1.00, 3, 0.100, 0.000, 0.0, 0.0,  90.0, 51.2410)
# scr3t = TeapotCollimatorNode(0.00450, 5, 1.00, 3, 0.100, 0.000, 0.0, 0.0,  45.0, 51.3902)
# scr3c = TeapotCollimatorNode(0.01455, 4, 1.00, 3, 0.100, 0.000, 0.0, 0.0,  45.0, 51.3947)
# scr4t = TeapotCollimatorNode(0.00450, 5, 1.00, 3, 0.100, 0.000, 0.0, 0.0, -45.0, 51.4346)
# scr4c = TeapotCollimatorNode(0.01455, 4, 1.00, 3, 0.100, 0.000, 0.0, 0.0, -45.0, 51.4391)
# p2sh1 = TeapotCollimatorNode(0.32000, 3, 1.00, 2, 0.100, 0.100, 0.0, 0.0,   0.0, 51.6228)
# p2    = TeapotCollimatorNode(1.80000, 3, 0.65, 2, 0.080, 0.048, 0.0, 0.0,   0.0, 51.9428)
# p2sh2 = TeapotCollimatorNode(0.32000, 3, 1.00, 2, 0.100, 0.100, 0.0, 0.0,   0.0, 53.7428)
# s1sh1 = TeapotCollimatorNode(1.01000, 3, 1.00, 1, 0.120, 0.000, 0.0, 0.0,   0.0, 57.2079)
# s1    = TeapotCollimatorNode(1.19000, 3, 0.65, 2, 0.0625, 0.0625, 0.0, 0.0, 0.0, 58.2179)
# s1sh2 = TeapotCollimatorNode(1.01000, 3, 1.00, 1, 0.120, 0.000, 0.0, 0.0,   0.0, 59.4079)
# s2sh1 = TeapotCollimatorNode(1.01000, 3, 1.00, 1, 0.120, 0.000, 0.0, 0.0,   0.0, 64.8679)
# s2    = TeapotCollimatorNode(1.19000, 3, 0.65, 2, 0.0625, 0.0625, 0.0, 0.0, 0.0, 65.8779)
# s2sh2 = TeapotCollimatorNode(1.01000, 3, 1.00, 1, 0.120, 0.000, 0.0, 0.0,   0.0, 67.0679)

# ap06200 = nodes[173]
# ap06201 = nodes[186]

# ap10000 = nodes[15]
# ap10001 = nodes[16]
# ap10002 = nodes[21]
# ap10003 = nodes[31]
# ap10004 = nodes[45]
# ap10005 = nodes[58]
# ap10006 = nodes[78]
# ap10007 = nodes[103]
# ap10008 = nodes[116]
# ap10009 = nodes[130]
# ap10010 = nodes[140]
# ap10011 = nodes[144]
# ap10012 = nodes[147]
# ap10013 = nodes[150]
# ap10014 = nodes[153]
# ap10015 = nodes[156]
# ap10016 = nodes[158]
# ap10017 = nodes[160]
# ap10018 = nodes[168]
# ap10019 = nodes[198]
# ap10020 = nodes[199]
# ap10021 = nodes[202]
# ap10022 = nodes[203]
# ap10023 = nodes[211]
# ap10024 = nodes[225]
# ap10025 = nodes[238]
# ap10026 = nodes[258]
# ap10027 = nodes[283]
# ap10028 = nodes[296]
# ap10029 = nodes[310]
# ap10030 = nodes[319]
# ap10031 = nodes[322]
# ap10032 = nodes[330]
# ap10033 = nodes[344]
# ap10034 = nodes[351]
# ap10035 = nodes[358]
# ap10036 = nodes[359]
# ap10037 = nodes[360]
# ap10038 = nodes[364]
# ap10039 = nodes[371]
# ap10040 = nodes[372]
# ap10041 = nodes[380]
# ap10042 = nodes[394]
# ap10043 = nodes[407]
# ap10044 = nodes[427]
# ap10045 = nodes[452]
# ap10046 = nodes[465]
# ap10047 = nodes[479]
# ap10048 = nodes[489]
# ap10049 = nodes[491]
# ap10050 = nodes[502]
# ap10051 = nodes[503]
# ap10052 = nodes[504]
# ap10053 = nodes[506]
# ap10054 = nodes[508]
# ap10055 = nodes[510]
# ap10056 = nodes[514]
# ap10057 = nodes[521]
# ap10058 = nodes[524]
# ap10059 = nodes[535]
# ap10060 = nodes[549]
# ap10061 = nodes[562]
# ap10062 = nodes[582]
# ap10063 = nodes[607]
# ap10064 = nodes[620]
# ap10065 = nodes[634]
# ap10066 = nodes[644]
# ap10067 = nodes[647]
# ap10068 = nodes[651]
# ap10069 = nodes[661]

# ap12000 = nodes[5]
# ap12001 = nodes[8]
# ap12002 = nodes[172]
# ap12003 = nodes[174]
# ap12004 = nodes[185]
# ap12005 = nodes[187]
# ap12006 = nodes[341]
# ap12007 = nodes[361]
# ap12008 = nodes[498]
# ap12009 = nodes[511]
# ap12010 = nodes[658]

# ap12500 = nodes[70]
# ap12501 = nodes[91]
# ap12502 = nodes[250]
# ap12503 = nodes[271]
# ap12504 = nodes[419]
# ap12505 = nodes[440]
# ap12506 = nodes[574]
# ap12507 = nodes[595]

# ap13000 = nodes[175]
# ap13001 = nodes[184]
# ap13002 = nodes[188]
# ap13003 = nodes[189]
# ap13004 = nodes[192]

# ap14000 = nodes[6]
# ap14001 = nodes[7]
# ap14002 = nodes[169]
# ap14003 = nodes[170]
# ap14004 = nodes[171]
# ap14005 = nodes[176]
# ap14006 = nodes[180]
# ap14007 = nodes[183]
# ap14008 = nodes[190]
# ap14009 = nodes[191]
# ap14010 = nodes[342]
# ap14011 = nodes[343]
# ap14012 = nodes[362]
# ap14013 = nodes[363]
# ap14014 = nodes[500]
# ap14015 = nodes[501]
# ap14016 = nodes[512]
# ap14017 = nodes[513]
# ap14018 = nodes[659]
# ap14019 = nodes[660]

# apell00 = nodes[35]
# apell01 = nodes[49]
# apell02 = nodes[62]
# apell03 = nodes[74]
# apell04 = nodes[87]
# apell05 = nodes[99]
# apell06 = nodes[112]
# apell07 = nodes[126]
# apell08 = nodes[159]
# apell09 = nodes[215]
# apell10 = nodes[229]
# apell11 = nodes[242]
# apell12 = nodes[254]
# apell13 = nodes[267]
# apell14 = nodes[279]
# apell15 = nodes[292]
# apell16 = nodes[306]
# apell17 = nodes[384]
# apell18 = nodes[398]
# apell19 = nodes[411]
# apell20 = nodes[423]
# apell21 = nodes[436]
# apell22 = nodes[448]
# apell23 = nodes[461]
# apell24 = nodes[475]
# apell25 = nodes[539]
# apell26 = nodes[553]
# apell27 = nodes[566]
# apell28 = nodes[578]
# apell29 = nodes[591]
# apell30 = nodes[603]
# apell31 = nodes[616]
# apell32 = nodes[630]

# ap06200.addChildNode(app06200, AccNode.EXIT)
# ap06201.addChildNode(app06201, AccNode.EXIT)

# ap10000.addChildNode(app10000, AccNode.EXIT)
# ap10001.addChildNode(app10001, AccNode.EXIT)
# ap10002.addChildNode(app10002, AccNode.EXIT)
# ap10003.addChildNode(app10003, AccNode.EXIT)
# ap10004.addChildNode(app10004, AccNode.EXIT)
# ap10005.addChildNode(app10005, AccNode.EXIT)
# ap10006.addChildNode(app10006, AccNode.EXIT)
# ap10007.addChildNode(app10007, AccNode.EXIT)
# ap10008.addChildNode(app10008, AccNode.EXIT)
# ap10009.addChildNode(app10009, AccNode.EXIT)
# ap10010.addChildNode(app10010, AccNode.EXIT)
# ap10011.addChildNode(app10011, AccNode.EXIT)
# ap10012.addChildNode(app10012, AccNode.EXIT)
# ap10013.addChildNode(app10013, AccNode.EXIT)
# ap10014.addChildNode(app10014, AccNode.EXIT)
# ap10015.addChildNode(app10015, AccNode.EXIT)
# ap10016.addChildNode(app10016, AccNode.EXIT)
# ap10017.addChildNode(app10017, AccNode.EXIT)
# ap10018.addChildNode(app10018, AccNode.EXIT)
# ap10019.addChildNode(app10019, AccNode.EXIT)
# ap10020.addChildNode(app10020, AccNode.EXIT)
# ap10021.addChildNode(app10021, AccNode.EXIT)
# ap10022.addChildNode(app10022, AccNode.EXIT)
# ap10023.addChildNode(app10023, AccNode.EXIT)
# ap10024.addChildNode(app10024, AccNode.EXIT)
# ap10025.addChildNode(app10025, AccNode.EXIT)
# ap10026.addChildNode(app10026, AccNode.EXIT)
# ap10027.addChildNode(app10027, AccNode.EXIT)
# ap10028.addChildNode(app10028, AccNode.EXIT)
# ap10029.addChildNode(app10029, AccNode.EXIT)
# ap10030.addChildNode(app10030, AccNode.EXIT)
# ap10031.addChildNode(app10031, AccNode.EXIT)
# ap10032.addChildNode(app10032, AccNode.EXIT)
# ap10033.addChildNode(app10033, AccNode.EXIT)
# ap10034.addChildNode(app10034, AccNode.EXIT)
# ap10035.addChildNode(app10035, AccNode.EXIT)
# ap10036.addChildNode(app10036, AccNode.EXIT)
# ap10037.addChildNode(app10037, AccNode.EXIT)
# ap10038.addChildNode(app10038, AccNode.EXIT)
# ap10039.addChildNode(app10039, AccNode.EXIT)
# ap10040.addChildNode(app10040, AccNode.EXIT)
# ap10041.addChildNode(app10041, AccNode.EXIT)
# ap10042.addChildNode(app10042, AccNode.EXIT)
# ap10043.addChildNode(app10043, AccNode.EXIT)
# ap10044.addChildNode(app10044, AccNode.EXIT)
# ap10045.addChildNode(app10045, AccNode.EXIT)
# ap10046.addChildNode(app10046, AccNode.EXIT)
# ap10047.addChildNode(app10047, AccNode.EXIT)
# ap10048.addChildNode(app10048, AccNode.EXIT)
# ap10049.addChildNode(app10049, AccNode.EXIT)
# ap10050.addChildNode(app10050, AccNode.EXIT)
# ap10051.addChildNode(app10051, AccNode.EXIT)
# ap10052.addChildNode(app10052, AccNode.EXIT)
# ap10053.addChildNode(app10053, AccNode.EXIT)
# ap10054.addChildNode(app10054, AccNode.EXIT)
# ap10055.addChildNode(app10055, AccNode.EXIT)
# ap10056.addChildNode(app10056, AccNode.EXIT)
# ap10057.addChildNode(app10057, AccNode.EXIT)
# ap10058.addChildNode(app10058, AccNode.EXIT)
# ap10059.addChildNode(app10059, AccNode.EXIT)
# ap10060.addChildNode(app10060, AccNode.EXIT)
# ap10061.addChildNode(app10061, AccNode.EXIT)
# ap10062.addChildNode(app10062, AccNode.EXIT)
# ap10063.addChildNode(app10063, AccNode.EXIT)
# ap10064.addChildNode(app10064, AccNode.EXIT)
# ap10065.addChildNode(app10065, AccNode.EXIT)
# ap10066.addChildNode(app10066, AccNode.EXIT)
# ap10067.addChildNode(app10067, AccNode.EXIT)
# ap10068.addChildNode(app10068, AccNode.EXIT)
# ap10069.addChildNode(app10069, AccNode.EXIT)

# ap12000.addChildNode(app12000, AccNode.EXIT)
# ap12001.addChildNode(app12001, AccNode.EXIT)
# ap12002.addChildNode(app12002, AccNode.EXIT)
# ap12003.addChildNode(app12003, AccNode.EXIT)
# ap12004.addChildNode(app12004, AccNode.EXIT)
# ap12005.addChildNode(app12005, AccNode.EXIT)
# ap12006.addChildNode(app12006, AccNode.EXIT)
# ap12007.addChildNode(app12007, AccNode.EXIT)
# ap12008.addChildNode(app12008, AccNode.EXIT)
# ap12009.addChildNode(app12009, AccNode.EXIT)
# ap12010.addChildNode(app12010, AccNode.EXIT)

# ap12500.addChildNode(app12500, AccNode.EXIT)
# ap12501.addChildNode(app12501, AccNode.EXIT)
# ap12502.addChildNode(app12502, AccNode.EXIT)
# ap12503.addChildNode(app12503, AccNode.EXIT)
# ap12504.addChildNode(app12504, AccNode.EXIT)
# ap12505.addChildNode(app12505, AccNode.EXIT)
# ap12506.addChildNode(app12506, AccNode.EXIT)
# ap12507.addChildNode(app12507, AccNode.EXIT)

# ap13000.addChildNode(app13000, AccNode.EXIT)
# ap13001.addChildNode(app13001, AccNode.EXIT)
# ap13002.addChildNode(app13002, AccNode.EXIT)
# ap13003.addChildNode(app13003, AccNode.EXIT)
# ap13004.addChildNode(app13004, AccNode.EXIT)

# ap14000.addChildNode(app14000, AccNode.EXIT)
# ap14001.addChildNode(app14001, AccNode.EXIT)
# ap14002.addChildNode(app14002, AccNode.EXIT)
# ap14003.addChildNode(app14003, AccNode.EXIT)
# ap14004.addChildNode(app14004, AccNode.EXIT)
# ap14005.addChildNode(app14005, AccNode.EXIT)
# ap14006.addChildNode(app14006, AccNode.EXIT)
# ap14007.addChildNode(app14007, AccNode.EXIT)
# ap14008.addChildNode(app14008, AccNode.EXIT)
# ap14009.addChildNode(app14009, AccNode.EXIT)
# ap14010.addChildNode(app14010, AccNode.EXIT)
# ap14011.addChildNode(app14011, AccNode.EXIT)
# ap14012.addChildNode(app14012, AccNode.EXIT)
# ap14013.addChildNode(app14013, AccNode.EXIT)
# ap14014.addChildNode(app14014, AccNode.EXIT)
# ap14015.addChildNode(app14015, AccNode.EXIT)
# ap14016.addChildNode(app14016, AccNode.EXIT)
# ap14017.addChildNode(app14017, AccNode.EXIT)
# ap14018.addChildNode(app14018, AccNode.EXIT)
# ap14019.addChildNode(app14019, AccNode.EXIT)

# apell00.addChildNode(appell00, AccNode.EXIT)
# apell01.addChildNode(appell01, AccNode.EXIT)
# apell02.addChildNode(appell02, AccNode.EXIT)
# apell03.addChildNode(appell03, AccNode.EXIT)
# apell04.addChildNode(appell04, AccNode.EXIT)
# apell05.addChildNode(appell05, AccNode.EXIT)
# apell06.addChildNode(appell06, AccNode.EXIT)
# apell07.addChildNode(appell07, AccNode.EXIT)
# apell08.addChildNode(appell08, AccNode.EXIT)
# apell09.addChildNode(appell09, AccNode.EXIT)
# apell10.addChildNode(appell10, AccNode.EXIT)
# apell11.addChildNode(appell11, AccNode.EXIT)
# apell12.addChildNode(appell12, AccNode.EXIT)
# apell13.addChildNode(appell13, AccNode.EXIT)
# apell14.addChildNode(appell14, AccNode.EXIT)
# apell15.addChildNode(appell15, AccNode.EXIT)
# apell16.addChildNode(appell16, AccNode.EXIT)
# apell17.addChildNode(appell17, AccNode.EXIT)
# apell18.addChildNode(appell18, AccNode.EXIT)
# apell19.addChildNode(appell19, AccNode.EXIT)
# apell20.addChildNode(appell20, AccNode.EXIT)
# apell21.addChildNode(appell21, AccNode.EXIT)
# apell22.addChildNode(appell22, AccNode.EXIT)
# apell23.addChildNode(appell23, AccNode.EXIT)
# apell24.addChildNode(appell24, AccNode.EXIT)
# apell25.addChildNode(appell25, AccNode.EXIT)
# apell26.addChildNode(appell26, AccNode.EXIT)
# apell27.addChildNode(appell27, AccNode.EXIT)
# apell28.addChildNode(appell28, AccNode.EXIT)
# apell29.addChildNode(appell29, AccNode.EXIT)
# apell30.addChildNode(appell30, AccNode.EXIT)
# apell31.addChildNode(appell31, AccNode.EXIT)
# apell32.addChildNode(appell32, AccNode.EXIT)

# addTeapotCollimatorNode(teapot_latt, 51.1921, scr1t)
# addTeapotCollimatorNode(teapot_latt, 51.1966, scr1c)
# addTeapotCollimatorNode(teapot_latt, 51.2365, scr2t)
# addTeapotCollimatorNode(teapot_latt, 51.2410, scr2c)
# addTeapotCollimatorNode(teapot_latt, 51.3902, scr3t)
# addTeapotCollimatorNode(teapot_latt, 51.3947, scr3c)
# addTeapotCollimatorNode(teapot_latt, 51.4346, scr4t)
# addTeapotCollimatorNode(teapot_latt, 51.4391, scr4c)


# Injection kickers
# -------------------------------------------------------------------------------
controller = inj.InjRegionController(
    ring,
    mass=mass,
    kin_energy=kin_energy,
    scale=10.0,
    inj_mid="injm1",
    inj_start="bpm_a09",
    inj_end="bpm_b01",
)

kicker_angles_t0, kicker_angles_t1 = [], []
# if _mpi_rank == 0:
    # This is just stalling when I run with more than one MPI node; not sure why.
    
#     # Bias the closed orbit.
#     print('Orbit bias:')
#     controller.set_inj_coords_vcorrectors([0.0, 0.0, 0.009, -0.0006], verbose=2)
#     controller.print_status()
    
#     # Set the initial phase space coordinates at the injection point.
#     solver_kws = dict(max_nfev=2500, verbose=2, ftol=1.0e-12, xtol=1.0e-12)
#     kicker_angles_t0 = controller.set_inj_coords(inj_coords_t0, **solver_kws)
#     controller.print_status()
    
#     # Set the final phase space coordinates at the injection point.
#     kicker_angles_t1 = controller.set_inj_coords(inj_coords_t1, **solver_kws)
#     controller.print_status()
    
#     # Convert to list for MPI_Bcast
#     kicker_angles_t0 = kicker_angles_t0.tolist()
#     kicker_angles_t1 = kicker_angles_t1.tolist()
    
# # Get kicker angles from rank 0
# kicker_angles_t0 = orbit_mpi.MPI_Bcast(kicker_angles_t0, orbit_mpi.mpi_datatype.MPI_DOUBLE, 0, _mpi_comm)
# kicker_angles_t1 = orbit_mpi.MPI_Bcast(kicker_angles_t1, orbit_mpi.mpi_datatype.MPI_DOUBLE, 0, _mpi_comm)
# kicker_angles_t0 = np.array(kicker_angles_t0)
# kicker_angles_t1 = np.array(kicker_angles_t1)

# # Initialize kicker angles.
# controller.set_kicker_angles(kicker_angles_t0)

# # Create the kicker waveforms
# ring.setLatticeOrder()
# t0 = 0.000
# t1 = n_inj_turns * seconds_per_turn
# amps_t0 = np.ones(8)
# amps_t1 = np.abs(kicker_angles_t1 / kicker_angles_t0)
# for node, a0, a1 in zip(controller.kicker_nodes, amps_t0, amps_t1):
#     waveform = SquareRootWaveform(t0=t0, t1=t1, a0=a0, a1=a1, sync_part=sync_part)
#     ring.setTimeDepNode(node.getParam('TPName'), waveform)


# Injection node and foil nodes
# ------------------------------------------------------------------------------

foil = {
    'name': 'foil node',
    'thickness': 390.0,
    'xmin': X_FOIL - 0.0085,
    'xmax': X_FOIL + 0.0085,
    'ymin': Y_FOIL - 0.0080,
    'ymax': Y_FOIL + 0.1000,
    'scatter': 0  # {0: full scatter, 1: simple scatter}
}
foil['node'] = TeapotFoilNode(
    foil['xmin'], 
    foil['xmax'], 
    foil['ymin'], 
    foil['ymax'], 
    foil['thickness'],
)
foil['node'].setScatterChoice(foil['scatter'])

injection_node = TeapotInjectionNode(
    nparts=n_macros_per_turn, 
    bunch=bunch, 
    lostbunch=lostbunch, 
    injectregion=[foil['xmin'], foil['xmax'], foil['ymin'], foil['ymax']], 
    xDistFunc=inj_dist['x']['gen'], 
    yDistFunc=inj_dist['y']['gen'], 
    lDistFun=inj_dist['z']['gen'], 
    nmaxmacroparticles=(n_macros_per_turn * n_inj_turns),
    name='inj node',
)
start_node = ring.getNodes()[0]
start_node.addChildNode(injection_node, AccNode.ENTRANCE)
start_node.addChildNode(foil['node'], AccNode.ENTRANCE)


# Black absorber collimator to act as an aperture
# ------------------------------------------------------------------------------

collimator = {
    'angle': 0.0,
    'density_fac': 1.0,
    'length': 0.00001,
    'ma': 9,
    'a': 0.110,  # radius
    'b': 0.0,
    'c': 0.0,
    'd': 0.0,
    'position': 0.5,
    'shape': 1,
    'name': 'collimator1',
}
collimator['node'] = TeapotCollimatorNode(
    collimator['length'], 
    collimator['ma'], 
    collimator['density_fac'], 
    collimator['shape'],
    collimator['a'], 
    collimator['b'], 
    collimator['c'], 
    collimator['d'], 
    collimator['angle'],
    collimator['position'], 
    collimator['name'],
)
addTeapotCollimatorNode(ring, collimator['position'], collimator['node'])


# RF cavities
#------------------------------------------------------------------------------
rf = dict()
rf = {
    'z_to_phi': 2.0 * np.pi / ring_length,
    'dE_sync': 0.0,
    'length': 0.0,
    'RF1a': {
        'phase': 0.0,
        'hnum': 1.0,
        'voltage': +5.03 * 1.0e-6,  # [GV]
        'name': 'RF1a',
        'position': 184.273,
    },
    'RF1b': {
        'phase': 0.0,
        'hnum': 1.0,
        'voltage': 0.0,  # [GV]
        'name': 'RF1b',
        'position': 186.571,
    },
    'RF1c': {
        'phase': 0.0,
        'hnum': 1.0,
        'voltage': 0.0,  # [GV]
        'name': 'RF1c',
        'position': 188.868,
    },
    'RF2': {
        'phase': 0.0,
        'hnum': 2.0,
        'voltage': -5.03 * 1.0e-6,  # [GV]
        'name': 'RF2',
        'position': 191.165,
    },
}

rf['RF1a']['node'] = RFNode.Harmonic_RFNode(
    rf['z_to_phi'], 
    rf['dE_sync'], 
    rf['RF1a']['hnum'], 
    rf['RF1a']['voltage'], 
    rf['RF1a']['phase'], 
    rf['length'], 
    rf['RF1a']['name'], 
)
rf['RF1b']['node'] = RFNode.Harmonic_RFNode(
    rf['z_to_phi'], 
    rf['dE_sync'], 
    rf['RF1b']['hnum'], 
    rf['RF1b']['voltage'], 
    rf['RF1b']['phase'], 
    rf['length'], 
    rf['RF1b']['name'], 
)
rf['RF1c']['node'] = RFNode.Harmonic_RFNode(
    rf['z_to_phi'], 
    rf['dE_sync'], 
    rf['RF1c']['hnum'], 
    rf['RF1c']['voltage'], 
    rf['RF1c']['phase'], 
    rf['length'], 
    rf['RF1c']['name'], 
)
rf['RF2']['node'] = RFNode.Harmonic_RFNode(
    rf['z_to_phi'], 
    rf['dE_sync'], 
    rf['RF2']['hnum'], 
    rf['RF2']['voltage'], 
    rf['RF2']['phase'], 
    rf['length'], 
    rf['RF2']['name'], 
)
RFLatticeModifications.addRFNode(ring, rf['RF1a']['position'], rf['RF1a']['node'])
RFLatticeModifications.addRFNode(ring, rf['RF1b']['position'], rf['RF1b']['node'])
RFLatticeModifications.addRFNode(ring, rf['RF1c']['position'], rf['RF1c']['node'])
RFLatticeModifications.addRFNode(ring, rf['RF2']['position'],  rf['RF2']['node'])



# Longitudinal impedance
# ------------------------------------------------------------------------------

impedance = dict()
impedance['longitudinal'] = {
    'phase_length': ring_length,
    'n_macros_min': 1000,
    'n_bins': 128,
    'position': 124.0,
}

# SNS Longitudinal Impedance tables. EKicker impedance from private communication 
# with J.G. Wang. Seems to be for 7 of the 14 kickers (not sure why). Impedance 
# units are [Ohms / n]. Kicker and RF impedances are inductive with real part 
# positive and imaginary part negative (Chao definition).
impedance['longitudinal']['ZL_EKicker'] = [
    complex(42.0, -182),
    complex(35, -101.5),
    complex(30.3333, -74.6667),
    complex(31.5, -66.5),
    complex(32.2, -57.4),
    complex(31.5, -51.333),
    complex(31, -49),
    complex(31.5, -46.375),
    complex(31.8889, -43.556),
    complex(32.9, -40.6),
    complex(32.7273, -38.18),
    complex(32.25, -35.58),
    complex(34.46, -32.846),
    complex(35, -30.5),
    complex(35.4667, -28.0),
    complex(36.75, -25.81),
    complex(36.647, -23.88),
    complex(36.944, -21.1667),
    complex(36.474, -20.263),
    complex(36.4, -18.55),
    complex(35.333, -17),
    complex(35, -14.95),
    complex(33.478, -13.69),
    complex(32.375, -11.67),
    complex(30.8, -10.08),
    complex(29.615, -8.077),
    complex(28.519, -6.74),
    complex(27.5, -5),
    complex(26.552, -4.103),
    complex(25.433, -3.266),
    complex(24.3871, -2.7),
    complex(23.40625, -2.18),
]
impedance['longitudinal']['ZL_RF'] = [
    complex(0.0, 0.0),
    complex(0.750, 0.0),
    complex(0.333, 0.0),
    complex(0.250, 0.0),
    complex(0.200, 0.0),
    complex(0.167, 0.0),
    complex(3.214, 0.0),
    complex(0.188, 0.0),
    complex(0.167, 0.0),
    complex(0.150, 0.0),
    complex(1.000, 0.0),
    complex(0.125, 0.0),
    complex(0.115, 0.0),
    complex(0.143, 0.0),
    complex(0.333, 0.0),
    complex(0.313, 0.0),
    complex(0.294, 0.0),
    complex(0.278, 0.0),
    complex(0.263, 0.0),
    complex(0.250, 0.0),
    complex(0.714, 0.0),
    complex(0.682, 0.0),
    complex(0.652, 0.0),
    complex(0.625, 0.0),
    complex(0.600, 0.0),
    complex(0.577, 0.0),
    complex(0.536, 0.0),
    complex(0.536, 0.0),
    complex(0.517, 0.0),
    complex(0.500, 0.0),
    complex(0.484, 0.0),
    complex(0.469, 0.0),
]
impedance['longitudinal']['Z'] = []
for zk, zrf in zip(impedance['longitudinal']['ZL_EKicker'], impedance['longitudinal']['ZL_RF']):
    zreal = (zk.real / 1.75) + zrf.real
    zimag = (zk.imag / 1.75) + zrf.imag
    impedance['longitudinal']['Z'].append(complex(zreal, zimag))

impedance['longitudinal']['node'] = LImpedance_Node(
    impedance['longitudinal']['phase_length'],
    impedance['longitudinal']['n_macros_min'],
    impedance['longitudinal']['n_bins'],
)
impedance['longitudinal']['node'].assignImpedance(impedance['longitudinal']['Z'])
addImpedanceNode(
    ring, 
    impedance['longitudinal']['position'], 
    impedance['longitudinal']['node'],
)


# Transverse impedance
# ------------------------------------------------------------------------------

impedance['transverse'] = {
    'phase_length': ring_length,
    'n_macros_min': 1000,
    'n_bins': 64,
    'position': 124.0,
    'use_x': 0,
    'use_y': 1,
    'q_x': 6.21991,
    'q_y': 6.20936,
    'alpha_x': 0.0,
    'alpha_y': -0.004,
    'beta_x': 10.191,
    'beta_y': 10.447,
}

# Read transverse impedance data from file.
impedance['transverse']['INDEX'] = []
impedance['transverse']['ZP'] = []
impedance['transverse']['ZM'] = []
file = open("_input/HahnImpedance.dat", "r")
for line in file:
    value = map(float, line.split())
    m = int(value[0])
    ZPR = value[1]
    ZPI = value[2]
    ZMR = value[3]
    ZMI = value[4]
    ZPAdd = complex(ZPR, -ZPI)
    ZMAdd = complex(ZMR, -ZMI)
    impedance['transverse']['INDEX'].append(m)
    impedance['transverse']['ZP'].append(ZPAdd)
    impedance['transverse']['ZM'].append(ZMAdd)
file.close()

# Write derived transverse impedance data to file.
file = open(get_filename('Hahn_impedance.dat'), "w")
for i in range(len(impedance['transverse']['ZP'])):
    file.write(
        '{} {} {}\n'.format(
            impedance['transverse']['INDEX'][i], 
            impedance['transverse']['ZP'][i], 
            impedance['transverse']['ZM'][i],
        ),
    )
file.close()

# Add transverse impedance node at lattice midpoint.
impedance['transverse']['node'] = TImpedance_Node(
    impedance['transverse']['phase_length'],
    impedance['transverse']['n_macros_min'], 
    impedance['transverse']['n_bins'], 
    impedance['transverse']['use_x'], 
    impedance['transverse']['use_y'],
)
impedance['transverse']['node'].assignLatFuncs(
    impedance['transverse']['q_x'],
    impedance['transverse']['alpha_x'],
    impedance['transverse']['beta_x'], 
    impedance['transverse']['q_y'],
    impedance['transverse']['alpha_y'],
    impedance['transverse']['beta_y'], 
)
if impedance['transverse']['use_x']:
    impedance['transverse']['node'].assignImpedance(
        "X", 
        impedance['transverse']['ZP'], 
        impedance['transverse']['ZM'],
    )
if impedance['transverse']['use_y']:
    impedance['transverse']['node'].assignImpedance(
        "Y", 
        impedance['transverse']['ZP'],
        impedance['transverse']['ZM'],
    )
addImpedanceNode(ring, impedance['transverse']['position'], impedance['transverse']['node'])


# Space charge
# ------------------------------------------------------------------------------

# Longitudinal space charge

space_charge = dict()
space_charge['longitudinal'] = {
    'b_a': (10.0 / 3.0),
    'length': ring_length,
    'use_spacecharge': 1,
    'n_bins': 128,
    'position': 124.0,
    'min_n_macros': 1000,
}
space_charge['longitudinal']['node'] = SC1D_AccNode(
    space_charge['longitudinal']['b_a'], 
    space_charge['longitudinal']['length'], 
    space_charge['longitudinal']['min_n_macros'], 
    space_charge['longitudinal']['use_spacecharge'], 
    space_charge['longitudinal']['n_bins'],
)
addLongitudinalSpaceChargeNode(
    ring, 
    space_charge['longitudinal']['position'], 
    space_charge['longitudinal']['node'],
)


# Transverse space charge

space_charge['transverse'] = {
    'constructor': SpaceChargeCalc2p5D,  # {SpaceChargeCalc2p5D, SpaceChargeCalcSliceBySlice2D}
    'node_setter': setSC2p5DAccNodes,  # {setSC2p5DAccNodes, setSC2DSliceBySliceAccNodes}
    'grid_size_x': 128,
    'grid_size_y': 128,
    'grid_size_z': 64,
    'path_length_min': 1.0e-8,
}
space_charge['transverse']['calc'] = space_charge['transverse']['constructor'](
    space_charge['transverse']['grid_size_x'],
    space_charge['transverse']['grid_size_y'],
    space_charge['transverse']['grid_size_z'],
)
boundary = {
    'n_boundary_points': 128,
    'n_free_space_modes': 32,
    'geometry': 'Circle',
    'radius': 0.22,
}
boundary['node'] = Boundary2D(
    boundary['n_boundary_points'], 
    boundary['n_free_space_modes'], 
    boundary['geometry'],
    boundary['radius'],
    boundary['radius'],
)
space_charge['transverse']['nodes'] = space_charge['transverse']['node_setter'](
    ring, 
    space_charge['transverse']['path_length_min'],
    space_charge['transverse']['calc'],
    boundary['node'],
)


# Diagnostics
# ------------------------------------------------------------------------------

# # Monitor node.
# bunch_monitor_node = BunchMonitorNode(
#     mm_mrad=True, 
#     transverse_only=False, 
#     skip=skip_inj
# )
# start_node.addChildNode(bunch_monitor_node, start_node.ENTRANCE)


# # Tune calculator
# tunes = TeapotTuneAnalysisNode("tune_analysis")
# tunes.assignTwiss(9.19025, -1.78574, -0.000143012, -2.26233e-05, 8.66549, 0.538244)
# addTeapotDiagnosticsNode(ring, 51.1921, tunes)


# Run simulation
# ------------------------------------------------------------------------------

def should_dump_bunch(turn):
    if  turn % 100 == 0:
        return True
    if turn == n_inj_turns + n_stored_turns - 1:
        return True
    return False


print("Tracking.")
for turn in trange(n_inj_turns + n_stored_turns):
    ring.trackBunch(bunch, params_dict)
    if should_dump_bunch(turn):
        bunch.dumpBunch("_output/data/bunch_turn{}.dat".format(turn))