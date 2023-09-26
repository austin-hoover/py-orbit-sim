from __future__ import print_function
import argparse
import os
import pickle
import sys
import time
from pprint import pprint

import numpy as np
import yaml
from tqdm import tqdm

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

import orbit_mpi
from bunch import Bunch
from orbit.bunch_utils import ParticleIdNumber
from orbit.injection import addTeapotInjectionNode
from orbit.injection import InjectParts
from orbit.injection import TeapotInjectionNode
from orbit.injection.joho import JohoLongitudinal
from orbit.injection.joho import JohoTransverse
from orbit.injection.distributions import SNSESpreadDist
from orbit.injection.distributions import UniformLongDist
from orbit.time_dep.waveform import ConstantWaveform
from orbit.time_dep.waveform import SquareRootWaveform
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccNode
from orbit.lattice import AccLattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice_Coupled
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_Ring
from orbit.utils import consts
from orbit.utils.consts import mass_proton
from orbit.utils.consts import speed_of_light

import pyorbit_sim

from sns_ring import SNS_RING
from ring_injection_controller import RingInjectionController


# Parse command line arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

# Settings
parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--madx-file", type=str, default="sns_ring_nux6.175_nuy6.175_sol.lattice")
parser.add_argument("--madx-seq", type=str, default="rnginjsol")
parser.add_argument("--save", type=int, default=1)
parser.add_argument("--save-ids", type=int, default=1)
parser.add_argument("--save-init-coords-attr", type=int, default=0)

# Lattice
parser.add_argument("--apertures", type=int, default=0)
parser.add_argument("--fringe", type=int, default=0)

parser.add_argument("--foil", type=int, default=0)
parser.add_argument("--foil-xmin", type=float, default=-0.0085)
parser.add_argument("--foil-xmax", type=float, default=+0.0085)
parser.add_argument("--foil-ymin", type=float, default=-0.0080)
parser.add_argument("--foil-ymax", type=float, default=+0.1000)
parser.add_argument("--foil-thick", type=float, default=390.0)
parser.add_argument("--foil-scatter", type=str, default="full", choices=["full", "simple"])

parser.add_argument("--imp-trans", type=int, default=0)
parser.add_argument("--imp-trans-n-macros-min", type=int, default=1000)
parser.add_argument("--imp-trans-n-bins", type=int, default=64)
parser.add_argument("--imp-trans-use-x", type=int, default=0)
parser.add_argument("--imp-trans-use-y", type=int, default=1)
parser.add_argument("--imp-trans-pos", type=float, default=124.0)
parser.add_argument("--imp-trans-alpha-x", type=float, default=0.0)
parser.add_argument("--imp-trans-alpha-y", type=float, default=-0.004)
parser.add_argument("--imp-trans-beta-x", type=float, default=10.191)
parser.add_argument("--imp-trans-beta-y", type=float, default=10.447)
parser.add_argument("--imp-trans-q-x", type=float, default=6.21991)
parser.add_argument("--imp-trans-q-y", type=float, default=6.20936)
parser.add_argument("--imp-trans-file", type=str, default="hahn_impedance.dat")

parser.add_argument("--imp-long", type=int, default=0)
parser.add_argument("--imp-long-n-macros-min", type=int, default=1000)
parser.add_argument("--imp-long-n-bins", type=int, default=128)
parser.add_argument("--imp-long-pos", type=float, default=124.0)
parser.add_argument("--imp-long-zl-ekick-file", type=str, default="zl_ekicker.dat")
parser.add_argument("--imp-long-zl-rf-file", type=str, default="zl_rf.dat")

parser.add_argument("--rf", type=int, default=0)
parser.add_argument("--rf1-phase", type=float, default=0.0)
parser.add_argument("--rf1-hnum", type=float, default=1.0)
parser.add_argument("--rf1-volt", type=float, default=+2.00e-06)
parser.add_argument("--rf2-phase", type=float, default=0.0)
parser.add_argument("--rf2-hnum", type=float, default=2.0)
parser.add_argument("--rf2-volt", type=float, default=-4.00e-06)

parser.add_argument("--sc", type=int, default=0)
parser.add_argument("--sc-kind", type=str, default="2p5d", choices=["slicebyslice", "2p5d"])
parser.add_argument("--sc-gridx", type=int, default=64)
parser.add_argument("--sc-gridy", type=int, default=64)
parser.add_argument("--sc-gridz", type=int, default=1)
parser.add_argument("--sc-path-length-min", type=float, default=1.00e-08)
parser.add_argument("--sc-n-macros-min", type=int, default=1000)
parser.add_argument("--sc-n-bound", type=int, default=128)
parser.add_argument("--sc-n-free", type=int, default=32)
parser.add_argument("--sc-radius", type=int, default=0.220)

parser.add_argument("--sc-long", type=int, default=0)
parser.add_argument("--sc-long-b-a", type=float, default=(10.0 / 3.0))
parser.add_argument("--sc-long-n-bins", type=int, default=64)
parser.add_argument("--sc-long-n-macros-min", type=int, default=1000)
parser.add_argument("--sc-long-pos", type=float, default=124.0)
parser.add_argument("--sc-long-use", type=int, default=1)

parser.add_argument("--sol", type=float, default=0.061)

# Bunch
parser.add_argument("--charge", type=float, default=1.0)
parser.add_argument("--energy", type=float, default=0.800)  # [GeV]
parser.add_argument("--mass", type=float, default=mass_proton)

parser.add_argument("--inj-x", type=float, default=0.0486)
parser.add_argument("--inj-y", type=float, default=0.0460)
parser.add_argument("--inj-dist-x", type=str, default="joho", choices=["joho"])
parser.add_argument("--inj-dist-y", type=str, default="joho", choices=["joho"])
parser.add_argument("--inj-dist-z", type=str, default="snsespread", choices=["snsespread", "uniform"])
parser.add_argument("--inj-dist-intensity", type=int, default=1.0)
parser.add_argument("--inj-dist-length", type=int, default=0.72)

# Injection
parser.add_argument("--x0", type=float, default=0.0)
parser.add_argument("--y0", type=float, default=0.0)
parser.add_argument("--xp0", type=float, default=0.0)
parser.add_argument("--yp0", type=float, default=0.0)

parser.add_argument("--x1", type=float, default=-0.030)
parser.add_argument("--y1", type=float, default=-0.030)
parser.add_argument("--xp1", type=float, default=0.0)
parser.add_argument("--yp1", type=float, default=0.0)

parser.add_argument("--inj-turns", type=int, default=300)
parser.add_argument("--stored-turns", type=int, default=0)

# Tracking/diagnostics
parser.add_argument("--n-parts", type=int, default=100000, help="Final number of macroparticles")
parser.add_argument("--vis-freq", type=int, default=0)
parser.add_argument("--write-bunch-freq", type=int, default=0)
parser.add_argument("--small-size", type=int, default=10000)
parser.add_argument("--small-freq", type=int, default=0)
parser.add_argument("--diag-tune", type=int, default=0)

args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

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
    
    
# Initialize bunch (needed for some aperture constructors).
# --------------------------------------------------------------------------------------

bunch = Bunch()
bunch.mass(args.mass)
bunch.charge(args.charge)
bunch.getSyncParticle().kinEnergy(args.energy)
lostbunch = Bunch()
lostbunch.addPartAttr("LostParticleAttributes")
params_dict = {"bunch": bunch, "lostbunch": lostbunch}
ring.set_bunch(bunch, lostbunch, params_dict)

# Compute the macroparticle size from the minipulse length, intensity, and number
# of macroparticles.
minipulse_intensity = args.inj_dist_intensity * config["minipulse_intensity"]
minipulse_intensity *= (args.inj_dist_length / config["minipulse_length"])
macros_per_turn = args.n_parts / args.inj_turns
macro_size = minipulse_intensity / macros_per_turn
bunch.macroSize(macro_size)


# Linear transfer matrix analysis (uncoupled)
# --------------------------------------------------------------------------------------

ring.set_fringe_fields(False)

test_bunch = Bunch()
test_bunch.mass(args.mass)
test_bunch.getSyncParticle().kinEnergy(args.energy)
matrix_lattice = TEAPOT_MATRIX_Lattice(ring, test_bunch)
tmat_params = matrix_lattice.getRingParametersDict()

if _mpi_rank == 0:
    logger.info("Transfer matrix parameters (2D):")
    for key, val in tmat_params.items():
        logger.info("{}: {}".format(key, val))

    if args.save:
        file = open(man.get_filename("lattice_params.pkl"), "wb")
        pickle.dump(tmat_params, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

twiss = ring.get_ring_twiss(mass=args.mass, kin_energy=args.energy)
dispersion = ring.get_ring_dispersion(mass=args.mass, kin_energy=args.energy)

if _mpi_rank == 0:
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

matrix_lattice = TEAPOT_MATRIX_Lattice_Coupled(ring, test_bunch, parameterization="LB")
tmat_params_4d = matrix_lattice.getRingParametersDict()

if _mpi_rank == 0:
    logger.info("Transfer matrix parameters (4D):")
    for key, val in tmat_params_4d.items():
        logger.info("{}: {}".format(key, val))

    if args.save:
        file = open(man.get_filename("lattice_params_4d.pkl"), "wb")
        pickle.dump(tmat_params_4d, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

twiss = ring.get_ring_twiss_coupled(mass=args.mass, kin_energy=args.energy)

if _mpi_rank == 0:
    logger.info("Twiss (coupled):")
    logger.info(twiss)
    if args.save:
        twiss.to_csv(man.get_filename("lattice_twiss_4d.dat"), sep=" ")

ring.set_fringe_fields(args.fringe)

            
# Injection kicker optimization
# --------------------------------------------------------------------------------------

# Initial closed orbit coordinates at injection point [x, x', y, y'].
inj_coords_t0 = np.zeros(4)
inj_coords_t0[0] = args.x0 + args.inj_x
inj_coords_t0[2] = args.y0 + args.inj_y
inj_coords_t0[1] = args.xp0
inj_coords_t0[3] = args.yp0

# Final closed orbit coordinates at injection point  [x, x', y, y'].
inj_coords_t1 = np.zeros(4)
inj_coords_t1[0] = args.x1 + args.inj_x
inj_coords_t1[2] = args.y1 + args.inj_y
inj_coords_t1[1] = args.xp1
inj_coords_t1[3] = args.yp1

ring.set_fringe_fields(False)

ric = ring.get_injection_controller(
    mass=args.mass, 
    kin_energy=args.energy, 
    inj_mid="injm1", 
    inj_start="bpm_a09", 
    inj_end="bpm_b01",
)
opt_kws = {
    "max_nfev": 2500,
    "verbose": 1, 
    "ftol": 1.00e-12, 
    "xtol": 1.00e-12,
    "gtol": 1.00e-12,
}

# ## Bias the vertical orbit using the vkickers.
# bias = False
# if bias:
#     print("Biasing vertical orbit using vkickers")
#     inj_controller.set_inj_coords_vcorrectors([0.0, 0.0, 0.007, -0.0005], verbose=1)
#     inj_controller.print_inj_coords()
# traj = inj_controller.get_trajectory()
# traj.to_csv(man.get_filename("inj_orbit_bias.dat"), sep=" ")

# Set the initial phase space coordinates at the injection point.
if _mpi_rank == 0:
    print("Optimizing kickers (t=0.0)")
kicker_angles_t0 = ric.set_inj_coords(inj_coords_t0, **opt_kws)
ric.print_inj_coords()
traj = ric.get_trajectory()
traj.to_csv(man.get_filename("inj_traj_t0.dat"), sep=" ")
if _mpi_rank == 0:
    logger.info("Injection region trajectory (t=0.0):")
    logger.info(traj)
    logger.info("Kicker angles:")
    logger.info(kicker_angles_t0)

# Set the final phase space coordinates at the injection point.
if _mpi_rank == 0:
    print("Optimizing kickers (t=1.0)")
kicker_angles_t1 = ric.set_inj_coords(inj_coords_t1, **opt_kws)
ric.print_inj_coords()
traj = ric.get_trajectory()
traj.to_csv(man.get_filename("inj_traj_t0.dat"), sep=" ")
if _mpi_rank == 0:
    logger.info("Injection region trajectory (t=0.0):")
    logger.info(traj)
    logger.info("Kicker angles:")
    logger.info(kicker_angles_t0)


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

ring.set_fringe_fields(args.fringe)

    
# Minipulse distribution
# --------------------------------------------------------------------------------------

inj_node = ring.add_inj_node(
    n_parts=macros_per_turn,
    n_parts_max=(macros_per_turn * args.inj_turns),
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
        "bunch_length_frac": args.inj_dist_length,
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
    
    
# Lattice
# --------------------------------------------------------------------------------------

if args.apertures:
    ring.add_inj_chicane_aperture_displacement_nodes()
    ring.add_collimator_nodes()
    # ring.add_aperture_nodes()  # not working
    
if args.foil:
    foil_node = ring.add_foil_node(
        xmin=args.foil_xmin,
        xmax=args.foil_xmax,
        ymin=args.foil_ymin,
        ymax=args.foil_ymax,
        thickness=args.foil_thick,
        scatter=args.foil_scatter,
    )

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

if args.imp_long:
    long_imp_node = ring.add_longitudinal_impedance_node(
        n_macros_min=args.imp_long_n_macros_min,
        n_bins=args.imp_long_n_bins,
        position=args.imp_long_pos,
        zl_ekicker_filename=os.path.join(input_dir, args.imp_long_zl_ekick_file),
        zl_rf_filename=os.path.join(input_dir, args.imp_long_zl_rf_file),
    )

if args.imp_trans:
    trans_imp_node = ring.add_transverse_impedance_node(
        n_macros_min=args.imp_trans_n_macros_min,
        n_bins=args.imp_trans_n_bins,
        use_x=args.imp_trans_use_x,
        use_y=args.imp_trans_use_y,
        position=args.imp_trans_pos,
        alpha_x=args.imp_trans_alpha_x,
        alpha_y=args.imp_trans_alpha_y,
        beta_x=args.imp_trans_beta_x,
        beta_y=args.imp_trans_beta_y,
        q_x=args.imp_trans_q_x,
        q_y=args.imp_trans_q_y,
        filename=os.path.join(input_dir, args.imp_trans_file)
    )

if args.sc_long:
    long_sc_node = ring.add_longitudinal_space_charge_node(
        b_a=args.sc_long_b_a,
        n_macros_min=args.sc_n_macros_min,
        use=args.sc_long_use,
        n_bins=args.sc_long_n_bins,
        position=args.sc_long_pos,
    )

if args.sc :
    trans_sc_nodes = ring.add_transverse_space_charge_nodes(
        n_macros_min=1000,
        size_x=args.sc_gridx,
        size_y=args.sc_gridy,
        size_z=args.sc_gridz,
        path_length_min=args.sc_path_length_min,
        n_boundary_points=args.sc_n_bound,
        n_free_space_modes=args.sc_n_free,
        radius=args.sc_radius,
        kind=args.sc_kind,
    )
    
    

# Diagnostics
# --------------------------------------------------------------------------------------

# Tune diagnostics node.
tune_node = ring.add_tune_diagnostics_node(
    filename=man.get_filename("tunes.dat"),
    position=0.010,
    alpha_x=tmat_params["alpha x"], 
    alpha_y=tmat_params["alpha y"],
    beta_x=tmat_params["beta x [m]"], 
    beta_y=tmat_params["beta y [m]"], 
    eta_x=tmat_params["dispersion x [m]"],
    etap_x=tmat_params["dispersion prime x"],
)   

# Plotting node
# [...]


# Tracking
# --------------------------------------------------------------------------------------

# monitor = pyorbit_sim.ring.Monitor(
#     filename=(man.get_filename("history.dat") if args.save else None),
#     verbose=True
# )

# if _mpi_rank == 0:
#     print("Tracking...")
    
# for turn in range(args.inj_turns + args.stored_turns):
#     ring.trackBunch(bunch, params_dict)  
#     monitor.action(params_dict)
#     if args.save and args.write_bunch_freq:
#         if (turn % args.write_bunch_freq == 0) or (turn == args.n_turns - 1):
#             filename = man.get_filename("bunch_{:05.0f}.dat".format(turn))
#             bunch.dumpBunch(filename)
            
#     if _mpi_rank == 0:
#         if args.save and args.small_freq:
#             if (turn % args.small_freq == 0) or (turn == args.n_turns - 1):
#                 X = pyorbit_sim.bunch_utils.get_coords(bunch, n=args.small_size)
#                 filename = "smallbunch_{:05.0f}.npy".format(turn)
#                 filename = man.get_filename(filename)
#                 np.save(filename, X)
                
                
#     if turn % 50 == 0:        
#         if _mpi_rank == 0:
#             X = np.zeros((bunch.getSize(), 6))
#             for i in range(X.shape[0]):
#                 X[i, :] = [
#                     bunch.x(i),
#                     bunch.xp(i), 
#                     bunch.y(i), 
#                     bunch.yp(i), 
#                     bunch.z(i), 
#                     bunch.dE(i)
#                 ]
#             X[:, :4] *= 1000.0
#             X[:, 5] *= 1000.0
                
#             dims = ["x", "xp", "y", "yp", "z", "dE"]
#             units = ["mm", "mrad", "mm", "mrad", "m", "MeV"]
#             dims_units = ["{} [{}]".format(dim, unit) for dim, unit in zip(dims, units)]
#             axis = (0, 2)
#             fig, ax = plt.subplots(figsize=(5.0, 5.0))
            
#             pad = 15.0
#             xmax = 1000.0 * ring.x_inj + pad
#             ymax = 1000.0 * ring.y_inj + pad
#             limits = [(-xmax, xmax), (-ymax, ymax)]
#             hist, edges = np.histogramdd(X[:, axis], bins=100, range=limits)
            
#             # hist = np.ma.masked_less_equal(hist, 0.0)
#             ax.pcolormesh(edges[0], edges[1], hist.T, cmap="viridis")
#             ax.set_xlabel(dims_units[axis[0]])
#             ax.set_xlabel(dims_units[axis[1]])
#             ax.set_xlim(limits[0])
#             ax.set_ylim(limits[1])
#             plt.savefig(man.get_filename("fig_proj2d_{:05.0f}.png".format(turn)), dpi=200)
                        

                

# if _mpi_rank == 0:
#     print("SIMULATION COMPLETE")
#     print("outdir = {}".format(man.outdir))
#     print("timestamp = {}".format(man.timestamp))
