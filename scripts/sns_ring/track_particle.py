"""Track a single particle through the SNS ring."""
from __future__ import print_function
import argparse
import math
import os
import pickle
import sys
import time
from pprint import pprint

import numpy as np
import yaml

from bunch import Bunch
from foil import Foil
from impedances import LImpedance
from impedances import TImpedance
from orbit.aperture import CircleApertureNode
from orbit.aperture import EllipseApertureNode
from orbit.aperture import RectangleApertureNode
from orbit.aperture import TeapotApertureNode
from orbit.aperture.ApertureLatticeModifications import addTeapotApertureNode
from orbit.bumps import bumps
from orbit.bumps import BumpLatticeModifications
from orbit.bumps import TDTeapotSimpleBumpNode
from orbit.bumps import TeapotBumpNode
from orbit.bumps import TeapotSimpleBumpNode
from orbit.collimation import addTeapotCollimatorNode
from orbit.collimation import TeapotCollimatorNode
from orbit.diagnostics import addTeapotDiagnosticsNode
from orbit.diagnostics import TeapotMomentsNode
from orbit.diagnostics import TeapotStatLatsNode
from orbit.diagnostics import TeapotTuneAnalysisNode
from orbit.foils import addTeapotFoilNode
from orbit.foils import TeapotFoilNode
from orbit.impedances import addImpedanceNode
from orbit.impedances import BetFreqDep_LImpedance_Node
from orbit.impedances import BetFreqDep_TImpedance_Node
from orbit.impedances import FreqDep_LImpedance_Node
from orbit.impedances import FreqDep_TImpedance_Node
from orbit.impedances import LImpedance_Node
from orbit.impedances import TImpedance_Node
from orbit.injection import addTeapotInjectionNode
from orbit.injection import InjectParts
from orbit.injection import TeapotInjectionNode
from orbit.injection.joho import JohoLongitudinal
from orbit.injection.joho import JohoTransverse
from orbit.injection.distributions import SNSESpreadDist
from orbit.injection.distributions import UniformLongDist
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccNode
from orbit.lattice import AccLattice
from orbit.rf_cavities import RFNode
from orbit.rf_cavities import RFLatticeModifications
from orbit.space_charge import sc2p5d
from orbit.space_charge import sc2dslicebyslice
from orbit.space_charge.sc1d import addLongitudinalSpaceChargeNode
from orbit.space_charge.sc1d import SC1D_AccNode
from orbit.space_charge.sc2dslicebyslice.scLatticeModifications import setSC2DSliceBySliceAccNodes
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice_Coupled
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_Ring
from orbit.time_dep.waveform import ConstantWaveform
from orbit.time_dep.waveform import SquareRootWaveform
from orbit.utils import consts
from orbit.utils.consts import mass_proton
from orbit.utils.consts import speed_of_light
import orbit_mpi
from spacecharge import LSpaceChargeCalc
from spacecharge import Boundary2D
from spacecharge import SpaceChargeCalc2p5D
from spacecharge import SpaceChargeCalcSliceBySlice2D

from sns_ring import SNS_RING

import pyorbit_sim


# Parse command line arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--madx-file", type=str, default="sns_ring_nux6.175_nuy6.175_sol.lattice")
parser.add_argument("--madx-seq", type=str, default="rnginjsol")
parser.add_argument("--save", type=int, default=1)

parser.add_argument("--apertures", type=int, default=0)
parser.add_argument("--fringe", type=int, default=0)
parser.add_argument("--foil", type=int, default=0)
parser.add_argument("--foil-thick", type=float, default=390.0)
parser.add_argument("--foil-scatter", type=str, default="full", choices=["full", "simple"])
parser.add_argument("--rf", type=int, default=0)
parser.add_argument("--rf1-phase", type=float, default=0.0)
parser.add_argument("--rf1-hnum", type=float, default=1.0)
parser.add_argument("--rf1-volt", type=float, default=+2.00e-06)
parser.add_argument("--rf2-phase", type=float, default=0.0)
parser.add_argument("--rf2-hnum", type=float, default=2.0)
parser.add_argument("--rf2-volt", type=float, default=-4.00e-06)
parser.add_argument("--sol", type=float, default=0.061, help="solenoid field [1/m]")

parser.add_argument("--charge", type=float, default=1.0)  # [elementary charge units]
parser.add_argument("--energy", type=float, default=1.0)  # [GeV]
parser.add_argument("--mass", type=float, default=mass_proton)  # [GeV / c^2]
parser.add_argument("--x", type=float, default=0.0)
parser.add_argument("--y", type=float, default=0.0)
parser.add_argument("--z", type=float, default=0.0)
parser.add_argument("--xp", type=float, default=0.0)
parser.add_argument("--yp", type=float, default=0.0)
parser.add_argument("--dE", type=float, default=0.0)

parser.add_argument("--n-turns", type=int, default=100)

parser.add_argument("--print_freq", type=int, default=1)

args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------
file_path = os.path.realpath(__file__)
file_dir = os.path.dirname(file_path)

# Load config file.
with open(os.path.join(file_dir, "config.yaml"), "r") as file:
    config = yaml.safe_load(file)

# Set input/output directories.
input_dir = os.path.join(file_dir, "data_input")
output_dir = os.path.join(file_dir, config["output_dir"])
if args.outdir is not None:
    output_dir = os.path.join(file_dir, args.outdir)

# Create output directory and save script info.
man = pyorbit_sim.utils.ScriptManager(outdir=output_dir, filepath=file_path)
if args.save:
    man.make_dirs()
    man.save_script_copy()
    
# Create logger.
logger = man.get_logger(save=args.save, disp=True)
for key, val in man.get_info().items():
    logger.info("{} {}".format(key, val))
logger.info(args)


# Initialize lattice
# --------------------------------------------------------------------------------------

ring = SNS_RING()
ring.readMADX(os.path.join(input_dir, args.madx_file), args.madx_seq)
ring.initialize()



ring.set_solenoid_strengths(args.sol)
    
    
# Initialize bunch
# --------------------------------------------------------------------------------------

bunch = Bunch()
bunch.mass(args.mass)
sync_part = bunch.getSyncParticle()
sync_part.kinEnergy(args.energy)
lostbunch = Bunch()
lostbunch.addPartAttr("LostParticleAttributes")
params_dict = {"bunch": bunch, "lostbunch": lostbunch}
ring.set_bunch(bunch, lostbunch, params_dict)

bunch.addParticle(args.x, args.xp, args.y, args.yp, args.z, args.dE)



# Linear transfer matrix analysis (uncoupled)
# --------------------------------------------------------------------------------------

ring.set_fringe_fields(False)

# Analyze the one-turn transfer matrix.
test_bunch = Bunch()
test_bunch.mass(args.mass)
test_bunch.getSyncParticle().kinEnergy(args.energy)
matrix_lattice = TEAPOT_MATRIX_Lattice(ring, test_bunch)
tmat_params = matrix_lattice.getRingParametersDict()

logger.info("Transfer matrix parameters (uncoupled):")
for key, val in tmat_params.items():
    logger.info("{}: {}".format(key, val))

if args.save:
    file = open(man.get_filename("lattice_params_uncoupled.pkl"), "wb")
    pickle.dump(tmat_params, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()

# Save the Twiss parameters throughout the ring.
twiss = ring.get_ring_twiss(mass=args.mass, kin_energy=args.energy)
dispersion = ring.get_ring_dispersion(mass=args.mass, kin_energy=args.energy)

logger.info("Twiss:")
logger.info(twiss)
if args.save:
    twiss.to_csv(man.get_filename("lattice_twiss.dat"), sep=" ")

logger.info("Dispersion:")
logger.info(dispersion)
if args.save:
    dispersion.to_csv(man.get_filename("lattice_dispersion.dat"), sep=" ")

ring.set_fringe_fields(args.fringe)


# Linear transfer matrix analysis (coupled)
# --------------------------------------------------------------------------------------

ring.set_fringe_fields(False)

# Analyze the one-turn transfer matrix.
matrix_lattice = TEAPOT_MATRIX_Lattice_Coupled(ring, test_bunch, parameterization="LB")
tmat_params = matrix_lattice.getRingParametersDict()

logger.info("Transfer matrix parameters (coupled):")
for key, val in tmat_params.items():
    logger.info("{}: {}".format(key, val))
    
if args.save:
    file = open(man.get_filename("lattice_params_coupled.pkl"), "wb")
    pickle.dump(tmat_params, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()

# Compute Twiss parameters throughout the ring. (This method does not propagate
# the Twiss parameters; it derives the parameters from the one-turn
# transfer matrix at each node.
twiss = ring.get_ring_twiss_coupled(mass=args.mass, kin_energy=args.energy)
logger.info("Twiss (coupled):")
logger.info(twiss)
if args.save:
    twiss.to_csv(man.get_filename("lattice_twiss_coupled.dat"), sep=" ")

ring.set_fringe_fields(args.fringe)


# Apertures
# --------------------------------------------------------------------------------------

if args.apertures:
    ring.add_inj_chicane_aperture_displacement_nodes()
    ring.add_collimator_nodes()
    ring.add_aperture_nodes()  # not working

if args.rf:
    rf_nodes = ring.add_harmonic_rf_node(
        rf1={
            "phase": args.rf1_phase,
            "hnum": args.rf1_hnum, 
            "voltage": args.rf1_volt,
        },
        rf2={
            "phase": args.rf2_phase, 
            "hnum": args.rf2_hnum, 
            "voltage": args.rf2_volt,
        },
    )


# Tracking
# --------------------------------------------------------------------------------------

coords = []
for turn in range(args.n_turns + 1):
    x = bunch.x(0)
    y = bunch.y(0)
    z = bunch.z(0)
    xp = bunch.xp(0)
    yp = bunch.yp(0)
    dE = bunch.dE(0)
    message = "turn={:05.0f} x={:<+7.3f} xp={:<+7.3f} y={:<+7.3f} yp={:<+7.3f} z={:<+7.3f} dE={:<+7.3f}".format(
        turn, 
        1000.0 * x, 
        1000.0 * xp, 
        1000.0 * y, 
        1000.0 * yp,
        z,
        dE,
    )
    if turn % args.print_freq == 0:
        logger.info(message)
    coords.append([x, xp, y, yp, z, dE])
    ring.trackBunch(bunch)

if args.save:
    filename = man.get_filename("coords.dat")
    print("Saving file {}".format(filename))
    np.savetxt(filename, coords)

print("SIMULATION COMPLETE")
if args.save:
    print("outdir = {}".format(man.outdir))
    print("timestamp = {}".format(man.timestamp))
