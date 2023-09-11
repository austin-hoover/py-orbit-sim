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


_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)


# Parse command line arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

# Settings
parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--madx-file", type=str, default="sns_ring_nux6.18_nuy6.18_dual_solenoid/lattice.lat")
parser.add_argument("--madx-seq", type=str, default="rnginjsol")
parser.add_argument("--save", type=int, default=1)
parser.add_argument("--save-init-coords-attr", type=int, default=0)

# Lattice
parser.add_argument("--apertures", type=int, default=0)
parser.add_argument("--fringe", type=int, default=0)
parser.add_argument("--foil", type=int, default=0)
parser.add_argument("--foil-thick", type=float, default=390.0)
parser.add_argument("--foil-scatter", type=str, default="full", choices=["full", "simple"])
parser.add_argument("--imp-trans", type=int, default=0)
parser.add_argument("--imp-long", type=float, default=0)
parser.add_argument("--rf", type=int, default=0)
parser.add_argument("--rf1-phase", type=float, default=0.0)
parser.add_argument("--rf1-hnum", type=float, default=1.0)
parser.add_argument("--rf1-volt", type=float, default=+2.00e-06)
parser.add_argument("--rf2-phase", type=float, default=0.0)
parser.add_argument("--rf2-hnum", type=float, default=2.0)
parser.add_argument("--rf2-volt", type=float, default=-4.00e-06)
parser.add_argument("--solenoid", type=int, default=0, help="Turns solenoid on/off.")
parser.add_argument("--sc", type=int, default=0)
parser.add_argument("--sc-long", type=int, default=0)
parser.add_argument("--sc-trans", type=str, default="2p5d", choices=["slicebyslice", "2p5d"])
parser.add_argument("--sc-gridx", type=int, default=64)
parser.add_argument("--sc-gridy", type=int, default=64)
parser.add_argument("--sc-gridz", type=int, default=64)

# Bunch
parser.add_argument("--bunch", type=str, default=None)
parser.add_argument("--charge", type=float, default=1.0)  # [elementary charge units]
parser.add_argument("--intensity", type=float, default=int(1.50e+14))
parser.add_argument("--energy", type=float, default=1.0)  # [GeV]
parser.add_argument("--mass", type=float, default=mass_proton)  # [GeV / c^2]
parser.add_argument("--bunch-length-frac", type=float, default=0.75, help="bunch length relative to ring")

# Initial/final injected phase space coordinates
parser.add_argument("--x0", type=float, default=0.0)
parser.add_argument("--y0", type=float, default=0.0)
parser.add_argument("--xp0", type=float, default=0.0)
parser.add_argument("--yp0", type=float, default=0.0)

parser.add_argument("--x1", type=float, default=25.0)
parser.add_argument("--y1", type=float, default=25.0)
parser.add_argument("--xp1", type=float, default=0.0)
parser.add_argument("--yp1", type=float, default=0.0)

# Injected minipulse distribution
parser.add_argument("--inj-x", type=float, default=0.0486)
parser.add_argument("--inj-y", type=float, default=0.0460)
parser.add_argument("--inj-dist-x", type=str, default="joho", choices=["joho"])
parser.add_argument("--inj-dist-y", type=str, default="joho", choices=["joho"])
parser.add_argument("--inj-dist-z", type=str, default="snsespread", choices=["snsespread", "uniform"])
parser.add_argument("--foil-xmin", type=float, default=-0.0085)
parser.add_argument("--foil-xmax", type=float, default=+0.0085)
parser.add_argument("--foil-ymin", type=float, default=-0.0080)
parser.add_argument("--foil-ymax", type=float, default=+0.1000)

# Diagnostics
parser.add_argument("--diag-tune", type=int, default=0)
parser.add_argument("--write_bunch_freq", type=int, default=100)

# Tracking
parser.add_argument("--macros-per-turn", type=int, default=1000)
parser.add_argument("--inj-turns", type=int, default=1000)
parser.add_argument("--stored-turns", type=int, default=0)

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
if args.save and _mpi_rank == 0:
    man.make_dirs()
    logger = man.get_logger(save=args.save, disp=True)
    for key, val in man.get_info().items():
        logger.info("{} {}".format(key, val))
    logger.info(args)
    man.save_script_copy()


# Create lattice
# --------------------------------------------------------------------------------------

# Create time-dependent lattice.
ring = SNS_RING(x_inj=args.inj_x, y_inj=args.inj_y)
ring.readMADX(os.path.join(input_dir, args.madx_file), args.madx_seq)
ring.initialize()

# Toggle solenoid.
for name in ["scbdsol_c13a", "scbdsol_c13b"]:
    node = ring.getNodeForName(name)
    B = 0.0
    if args.solenoid:
        B = 0.6 / (2.0 * node.getLength())
    node.setParam("B", B)
    if _mpi_rank == 0:
        logger.info(
            "{}: B={:.2f}, L={:.2f}".format(
                node.getName(), 
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
if _mpi_rank == 0:
    logger.info("Transfer matrix parameters (uncoupled):")
    logger.info(tmat_params)
file = open(man.get_filename("lattice_params_uncoupled.pkl"), "wb")
pickle.dump(tmat_params, file, protocol=pickle.HIGHEST_PROTOCOL)
file.close()

# Save the Twiss parameters throughout the ring.
twiss = ring.get_ring_twiss(mass=args.mass, kin_energy=args.energy)
dispersion = ring.get_ring_dispersion(mass=args.mass, kin_energy=args.energy)
if _mpi_rank == 0:
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
if _mpi_rank == 0:
    logger.info("Transfer matrix parameters (coupled):")
    logger.info(tmat_params)
file = open(man.get_filename("lattice_params_coupled.pkl"), "wb")
pickle.dump(tmat_params, file, protocol=pickle.HIGHEST_PROTOCOL)
file.close()

# Compute Twiss parameters throughout the ring. (This method does not propagate
# the Twiss parameters; it derives the parameters from the one-turn
# transfer matrix at each node.
twiss_coupled = ring.get_ring_twiss_coupled(mass=args.mass, kin_energy=args.energy)
if _mpi_rank == 0:
    logger.info("Twiss:")
    logger.info(twiss_coupled)
    twiss.to_csv(man.get_filename("lattice_twiss_coupled.dat"), sep=" ")

ring.set_fringe_fields(args.fringe)


# Bunch setup
# --------------------------------------------------------------------------------------

# Initialize the bunch.
bunch = Bunch()
bunch.mass(args.mass)
sync_part = bunch.getSyncParticle()
sync_part.kinEnergy(args.energy)
lostbunch = Bunch()
lostbunch.addPartAttr("LostParticleAttributes")
params_dict = {"bunch": bunch, "lostbunch": lostbunch}
ring.set_bunch(bunch, lostbunch, params_dict)

# Compute the macroparticle size from the minipulse length, intensity, and number
# of macroparticles.
bunch_length_factor = args.bunch_length_frac / config["bunch_length_frac"]
minipulse_intensity = config["minipulse_intensity"] * bunch_length_factor
intensity = minipulse_intensity * args.inj_turns
macro_size = intensity / args.inj_turns / args.macros_per_turn
bunch.macroSize(macro_size)

            
# Injection kicker optimization
# --------------------------------------------------------------------------------------

# Initial closed orbit coordinates at injection point [x, x', y, y'].
inj_coords_t0 = np.zeros(4)
inj_coords_t0[0] = args.inj_x - args.x0
inj_coords_t0[1] = 0.0 - args.xp0
inj_coords_t0[2] = args.inj_y - args.y0
inj_coords_t0[3] = 0.0 - args.yp0

# Final closed orbit coordinates at injection point  [x, x', y, y'].
inj_coords_t1 = np.zeros(4)
inj_coords_t1[0] = args.inj_x - args.x1
inj_coords_t1[1] = 0.000 - args.xp1
inj_coords_t1[2] = args.inj_y - args.y1
inj_coords_t1[3] = 0.000 - args.yp1

# # Fringe fields complicate things. Turn them off for now.
# ring.set_fringe_fields(False)

# inj_controller = ring.get_injection_controller(
#     mass=mass, 
#     kin_energy=kin_energy, 
#     inj_mid="injm1", 
#     inj_start="bpm_a09", 
#     inj_end="bpm_b01",
# )
# inj_controller.scale_kicker_limits(100.0)
# solver_kws = dict(max_nfev=2500, verbose=1, ftol=1.0e-12, xtol=1.0e-12)

# ## Bias the vertical orbit using the vkickers.
# bias = False
# if bias:
#     print("Biasing vertical orbit using vkickers")
#     inj_controller.set_inj_coords_vcorrectors([0.0, 0.0, 0.007, -0.0005], verbose=1)
#     inj_controller.print_inj_coords()
# traj = inj_controller.get_trajectory()
# traj.to_csv(man.get_filename("inj_orbit_bias.dat"), sep=" ")

# ## Set the initial phase space coordinates at the injection point.
# print("Optimizing kickers (t=0)")
# kicker_angles_t0 = inj_controller.set_inj_coords_fast(inj_coords_t0, **solver_kws)
# inj_controller.print_inj_coords()
# traj = inj_controller.get_trajectory()
# traj.to_csv(man.get_filename("inj_orbit_t0.dat"), sep=" ")

# ## Set the final phase space coordinates at the injection point.
# print("Optimizing kickers (t=1)")
# kicker_angles_t1 = inj_controller.set_inj_coords_fast(inj_coords_t1, **solver_kws)
# inj_controller.print_inj_coords()
# traj = inj_controller.get_trajectory()
# traj.to_csv(man.get_filename("inj_orbit_t1.dat"), sep=" ")

# # Set kicker waveforms.
# seconds_per_turn = ring.getLength() / (sync_part.beta() * consts.speed_of_light)
# t0 = 0.0
# t1 = n_inj_turns * seconds_per_turn
# strengths_t0 = np.ones(8)
# strengths_t1 = np.abs(kicker_angles_t1 / kicker_angles_t0)
# for node, s0, s1 in zip(inj_controller.kicker_nodes, strengths_t0, strengths_t1):
#     waveform = NthRootWaveform(n=2.0, t0=t0, t1=t1, s0=s0, s1=s1, sync_part=sync_part)
#     # node.setWaveform(waveform)

# # Set kickers to t0 state.
# inj_controller.set_kicker_angles(kicker_angles_t0)

# # Toggle fringe fields (see previous above).
# ring.set_fringe_fields(switches["fringe"])

    
# Injection
# --------------------------------------------------------------------------------------

inj_node = ring.add_inj_node(
    n_parts=args.macros_per_turn,
    n_parts_max=(args.macros_per_turn * args.inj_turns),
    xmin=args.foil_xmin,
    xmax=args.foil_xmax,
    ymin=args.foil_ymin,
    ymax=args.foil_ymax,
    dist_x=args.inj_dist_x,
    dist_y=args.inj_dist_y,
    dist_z=args.inj_dist_z,
    dist_x_kws={
        "centerpos": args.inj_x,
        "centermom": 0.0,
        "eps_rms": 0.221e-06,
    },
    dist_y_kws={
        "centerpos": args.inj_y,
        "centermom": 0.0,
        "eps_rms": 0.221e-06,
    },
    dist_z_kws={
        "n_inj_turns": args.inj_turns,
        "bunch_length_frac": args.bunch_length_frac,
    }
)

if args.foil:
    foil_node = ring.add_foil_node(
        xmin=args.foil_xmin,
        xmax=args.foil_xmax,
        ymin=args.foil_ymin,
        ymax=args.foil_ymax,
        thickness=args.foil_thick,
        scatter=args.foil_scatter,
    )


# Apertures, collimators, and displacements
# --------------------------------------------------------------------------------------

if args.apertures:
    ring.add_inj_chicane_aperture_displacement_nodes()
    ring.add_collimator_nodes()
    # ring.add_aperture_nodes()  # not working


# RF cavities
# --------------------------------------------------------------------------------------

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


# Impedance
# --------------------------------------------------------------------------------------

if args.imp_long:
    long_imp_node = ring.add_longitudinal_impedance_node(
        n_macros_min=1000,
        n_bins=128,
        position=124.0,
        zl_ekicker_filename=os.path.join(input_dir, "zl_ekicker.dat"),
        zl_rf_filename=os.path.join(input_dir, "zl_rf.dat"),
    )

if args.imp_trans:
    trans_imp_node = ring.add_transverse_impedance_node(
        n_macros_min=1000,
        n_bins=64,
        use_x=0,
        use_y=1,
        position=124.0,
        alpha_x=0.0,
        alpha_y=-0.004,
        beta_x=10.191,
        beta_y=10.447,
        q_x=6.21991,
        q_y=6.20936,
        filename=os.path.join(input_dir, "hahn_impedance.dat"),
    )


# Space charge
# --------------------------------------------------------------------------------------

if args.sc and args.sc_long:
    long_sc_node = ring.add_longitudinal_space_charge_node(
        b_a=(10.0 / 3.0),
        n_macros_min=1000,
        use=1,
        n_bins=64,
        position=124.0,
    )

if args.sc:
    trans_sc_nodes = ring.add_transverse_space_charge_nodes(
        n_macros_min=1000,
        size_x=args.sc_gridx,
        size_y=args.sc_gridy,
        size_z=args.sc_gridz,
        path_length_min=1.00e-08,
        n_boundary_points=128,
        n_free_space_modes=32,
        radius=0.220,
        kind=args.sc_trans,
    )


# Diagnostics
# --------------------------------------------------------------------------------------

if args.diag_tune:
    index = 0  # not much dispersion at injection?
    tune_node = ring.add_tune_diagnostics_node(
        filename=man.get_filename("tunes.dat"),
        position=twiss.loc[index, "s"],
        beta_x=twiss.loc[index, "beta_x"],
        beta_y=twiss.loc[index, "beta_y"],
        alpha_x=twiss.loc[index, "alpha_x"],
        alpha_y=twiss.loc[index, "alpha_y"],
        eta_x=dispersion.loc[index, "disp_x"],
        etap_x=dispersion.loc[index, "dispp_x"],
    )


# Tracking
# --------------------------------------------------------------------------------------

n_turns = args.inj_turns + args.stored_turns

monitor = pyorbit_sim.ring.Monitor(
    filename=man.get_filename("history.dat"), 
    verbose=True
)

if _mpi_rank == 0:
    print("Tracking...")
    
for turn in range(n_turns):
    ring.trackBunch(bunch, params_dict)  
    monitor.action(params_dict)
    if (turn % args.write_bunch_freq == 0) or (turn == n_turns - 1):
        filename = man.get_filename("bunch_{:05.0f}.dat".format(turn))
        bunch.dumpBunch(filename)
        
if _mpi_rank == 0:
    print("SIMULATION COMPLETE")
    print("outdir = {}".format(man.outdir))
    print("timestamp = {}".format(man.timestamp))