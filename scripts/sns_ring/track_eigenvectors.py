"""Track a single particle through the SNS ring."""
from __future__ import print_function
import argparse
import math
import os
import pickle
from pprint import pprint
import sys
import time

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

# Settings
parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--madx-file", type=str, default="sns_ring_nux6.18_nuy6.18_dual_solenoid/lattice.lat")
parser.add_argument("--madx-seq", type=str, default="rnginjsol")
parser.add_argument("--save", type=int, default=1)

# Lattice
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
parser.add_argument("--solenoid", type=int, default=0, help="Turns solenoid on/off.")

# Bunch
parser.add_argument("--charge", type=float, default=1.0)  # [elementary charge units]
parser.add_argument("--intensity", type=float, default=int(1.50e+14))
parser.add_argument("--energy", type=float, default=1.0)  # [GeV]
parser.add_argument("--mass", type=float, default=mass_proton)  # [GeV / c^2]
parser.add_argument("--bunch-length-frac", type=float, default=0.75, help="bunch length relative to ring")

parser.add_argument("--n-turns", type=int, default=100)

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
    logger = man.get_logger(save=args.save, disp=True)
    for key, val in man.get_info().items():
        logger.info("{} {}".format(key, val))
    logger.info(args)
    man.save_script_copy()


# Initialize lattice
# --------------------------------------------------------------------------------------

ring = SNS_RING()
ring.readMADX(os.path.join(input_dir, args.madx_file), args.madx_seq)
ring.initialize()

for name in ["scbdsol_c13a", "scbdsol_c13b"]:
    node = ring.getNodeForName(name)
    B = 1.00e-12
    if args.solenoid:
        B = 0.6 / (2.0 * node.getLength())
    node.setParam("B", B)
    logger.info(
        "{}: type={}, B={:.2f}, L={:.2f}".format(
            node.getName(), 
            type(node),
            node.getParam("B"),
            node.getLength())
    )

    

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
logger.info(tmat_params)

file = open(man.get_filename("lattice_params_uncoupled.pkl"), "wb")
pickle.dump(tmat_params, file, protocol=pickle.HIGHEST_PROTOCOL)
file.close()

# Save the Twiss parameters throughout the ring.
twiss = ring.get_ring_twiss(mass=args.mass, kin_energy=args.energy)
dispersion = ring.get_ring_dispersion(mass=args.mass, kin_energy=args.energy)

logger.info("Twiss:")
logger.info(twiss)
twiss.to_csv(man.get_filename("lattice_twiss.dat"), sep=" ")
logger.info("Dispersion:")
logger.info(dispersion)
dispersion.to_csv(man.get_filename("lattice_dispersion.dat"), sep=" ")

ring.set_fringe_fields(args.fringe)


# Linear transfer matrix analysis (coupled)
# --------------------------------------------------------------------------------------

ring.set_fringe_fields(False)

# Analyze the one-turn transfer matrix.
matrix_lattice = TEAPOT_MATRIX_Lattice_Coupled(ring, test_bunch, parameterization="LB")
tmat_params = matrix_lattice.getRingParametersDict()

logger.info("Transfer matrix parameters (coupled):")
logger.info(tmat_params)

file = open(man.get_filename("lattice_params_coupled.pkl"), "wb")
pickle.dump(tmat_params, file, protocol=pickle.HIGHEST_PROTOCOL)
file.close()

# Compute Twiss parameters throughout the ring. (This method does not propagate
# the Twiss parameters; it derives the parameters from the one-turn
# transfer matrix at each node.
twiss = ring.get_ring_twiss_coupled(mass=args.mass, kin_energy=args.energy)

logger.info("Twiss (coupled):")
logger.info(twiss)
twiss.to_csv(man.get_filename("lattice_twiss_coupled.dat"), sep=" ")

ring.set_fringe_fields(args.fringe)


# Apertures
# --------------------------------------------------------------------------------------

bunch = Bunch()
bunch.mass(args.mass)
sync_part = bunch.getSyncParticle()
sync_part.kinEnergy(args.energy)
lostbunch = Bunch()
lostbunch.addPartAttr("LostParticleAttributes")
params_dict = {"bunch": bunch, "lostbunch": lostbunch}
ring.set_bunch(bunch, lostbunch, params_dict)

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

# Add particles along each eigenvector.
eigenvectors = tmat_params["eigenvectors"]
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
    
# Track the bunch.
coords = [[], []] # n_modes x n_turns x 6
for turn in range(args.n_turns + 1):
    for mode in range(2):
        x = bunch.x(mode)
        y = bunch.y(mode)
        z = bunch.z(mode)
        xp = bunch.xp(mode)
        yp = bunch.yp(mode)
        dE = bunch.dE(mode)
        coords[mode].append([x, xp, y, yp, z, dE])
        print(
            "turn={:05.0f} mode={} x={:<7.3f} xp={:<7.3f} y={:<7.3f} yp={:<7.3f} z={:<7.3f} dE={:<7.3f}".format(
                turn, 
                mode, 
                1000.0 * x, 
                1000.0 * xp, 
                1000.0 * y, 
                1000.0 * yp,
                z,
                dE,
            )
        )
    ring.trackBunch(bunch)

# Save TBT coordinates.
for i in range(len(coords)):
    coords[i] = np.array(coords[i])
if args.save:
    for i in range(len(coords)):
        mode = i + 1
        filename = man.get_filename("coords_mode{}.dat".format(mode))
        print("Saving file {}".format(filename))
        np.savetxt(filename, coords[i]) 

print("SIMULATION COMPLETE")
print("outdir = {}".format(man.outdir))
print("timestamp = {}".format(man.timestamp))
