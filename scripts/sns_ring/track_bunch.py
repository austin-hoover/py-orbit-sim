"""Track a bunch through the SNS ring.

Currently, the script requires the bunch to be read from a file. A future version 
will allow the use to generate a bunch from a distribution function.
"""
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

import orbit_mpi
from bunch import Bunch
from orbit.bunch_generators import KVDist2D
from orbit.bunch_generators import GaussDist1D
from orbit.bunch_generators import GaussDist2D
from orbit.bunch_generators import WaterBagDist2D
from orbit.bunch_utils import ParticleIdNumber
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccNode
from orbit.lattice import AccLattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice_Coupled
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_Ring
from orbit.utils.consts import mass_proton
from orbit.utils.consts import speed_of_light

import pyorbit_sim

from sns_ring import SNS_RING
from bunch_generator import BunchGenerator


# Parse command line arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

# Settings
parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--madx-file", type=str, default="sns_ring_nux6.175_nuy6.175_sol.lattice")
parser.add_argument("--madx-seq", type=str, default="rnginjsol")
parser.add_argument("--save", type=int, default=1)
parser.add_argument("--save-ids", type=int, default=1)

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
parser.add_argument("--bunch", type=str, default=None)
parser.add_argument("--charge", type=float, default=1.0)
parser.add_argument("--energy", type=float, default=0.800)  # [GeV]
parser.add_argument("--mass", type=float, default=mass_proton)
parser.add_argument("--intensity", type=float, default=0.45e+14)

parser.add_argument("--mean-x", type=float, default=None)
parser.add_argument("--mean-y", type=float, default=None)
parser.add_argument("--mean-z", type=float, default=None)
parser.add_argument("--mean-xp", type=float, default=None)
parser.add_argument("--mean-yp", type=float, default=None)
parser.add_argument("--mean-dE", type=float, default=None)

parser.add_argument("--n-parts", type=int, default=None)
parser.add_argument("--dist", type=str, default="waterbag", choices=["waterbag", "gaussian", "kv"])
parser.add_argument("--coast", type=int, default=0)
parser.add_argument("--match", type=int, default=0)
parser.add_argument("--rms-equiv", type=int, default=0)

parser.add_argument("--n-turns", type=int, default=100)
parser.add_argument("--print-freq", type=int, default=1)
parser.add_argument("--vis-freq", type=int, default=0)
parser.add_argument("--write-bunch-freq", type=int, default=0)

parser.add_argument("--small-size", type=int, default=10000)
parser.add_argument("--small-freq", type=int, default=0)

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
    
    
# Bunch
# --------------------------------------------------------------------------------------

filename = args.bunch

dist_constructors = {
    "kv": KVDist2D, 
    "waterbag": WaterBagDist2D, 
    "gaussian": GaussDist2D,
}
dist_constructor = dist_constructors[args.dist]

n_parts = args.n_parts
if n_parts is None:
    n_parts = 100000

    
if filename is None:
    bunch = pyorbit_sim.bunch_utils.generate_xy_z(
        dist_xy=dist_constructor(), 
        dist_z=GaussDist1D(), 
        n=n_parts, 
        bunch=bunch,
        verbose=True
    )
    z_max = 0.5 * ring.getLength()
    for i in range(bunch.getSize()):
        bunch.z(i, np.random.uniform(-z_max, z_max))
        bunch.dE(i, 1.00e-12)
    pyorbit_sim.ring.match_bunch(bunch, M=tmat_params_4d["M"][:4, :4])
else:
    bunch = pyorbit_sim.bunch_utils.load(filename=filename, bunch=bunch, verbose=True)

bunch = pyorbit_sim.bunch_utils.set_centroid(
    bunch,
    centroid=[
        args.mean_x, 
        args.mean_y, 
        args.mean_z, 
        args.mean_xp, 
        args.mean_xp, 
        args.mean_dE,
    ],
    verbose=True,
)

if args.rms_equiv:
    coords_z = [(bunch.z(i), bunch.dE(i)) for i in range(bunch.getSize())]
    bunch = pyorbit_sim.bunch_utils.generate_rms_equiv_xy_z(
        dist_constructor_xy=dist_constructor,
        dist_constructor_z=GaussDist1D,
        bunch=bunch,
        verbose=True,
    )
    for i in range(bunch.getSize()):
        bunch.z(i, coords_z[i][0])
        bunch.dE(i, coords_z[i][0])

if args.n_parts:
    bunch = pyorbit_sim.bunch_utils.downsample(
        bunch,
        n=args.n_parts,
        method="first",
        conserve_intensity=True,
        verbose=True,
    )
    
if args.coast:
    z_max = 0.5 * ring.getLength()
    for i in range(bunch.getSize()):
        bunch.z(i, np.random.uniform(-z_max, z_max))
        bunch.dE(i, 1.00e-12)
    
if args.match:
    bunch = pyorbit_sim.ring.match_bunch(bunch, M=tmat_params_4d["M"][:6, :6])

bunch_size_global = bunch.getSizeGlobal()
macro_size = args.intensity / bunch_size_global
bunch.macroSize(macro_size)

if args.save_ids:
    ParticleIdNumber.addParticleIdNumbers(bunch)


# Diagnostics
# --------------------------------------------------------------------------------------

# Plotting node
# [...]


# Tracking
# --------------------------------------------------------------------------------------

monitor = pyorbit_sim.ring.Monitor(
    filename=(man.get_filename("history.dat") if args.save else None),
    verbose=True
)

if _mpi_rank == 0:
    print("Tracking...")
    
for turn in range(args.n_turns):
    ring.trackBunch(bunch, params_dict)  
    monitor.action(params_dict)
    if args.save and args.write_bunch_freq:
        if (turn % args.write_bunch_freq == 0) or (turn == args.n_turns - 1):
            filename = man.get_filename("bunch_{:05.0f}.dat".format(turn))
            bunch.dumpBunch(filename)
            
    if _mpi_rank == 0:
        if args.save and args.small_freq:
            if (turn % args.small_freq == 0) or (turn == args.n_turns - 1):
                X = pyorbit_sim.bunch_utils.get_coords(bunch, n=args.small_size)
                filename = "smallbunch_{:05.0f}.npy".format(turn)
                filename = man.get_filename(filename)
                np.save(filename, X)

        
if _mpi_rank == 0:
    print("SIMULATION COMPLETE")
    print("outdir = {}".format(man.outdir))
    print("timestamp = {}".format(man.timestamp))
