from __future__ import print_function
import math
import os
import pathlib
from pprint import pprint
import shutil
import sys
import time

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from orbit.diagnostics.diagnostics import BunchCoordsNode
from orbit.diagnostics.diagnostics_lattice_modifications import add_diagnostics_node_as_child
from orbit.diagnostics import addTeapotDiagnosticsNode
from orbit.diagnostics import TeapotMomentsNode
from orbit.diagnostics import TeapotStatLatsNode
from orbit.diagnostics import TeapotTuneAnalysisNode
from orbit.envelope import DanilovEnvelope
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
from orbit.injection.distributions import JohoLongitudinal
from orbit.injection.distributions import JohoTransverse
from orbit.injection.distributions import SNSESpreadDist
from orbit.injection.distributions import UniformLongDist
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccNode
from orbit.lattice import AccLattice
from orbit.rf_cavities import RFNode
from orbit.rf_cavities import RFLatticeModifications
from orbit.space_charge import sc2p5d
from orbit.space_charge import sc2dslicebyslice
from orbit.space_charge.envelope import set_env_solver_nodes
from orbit.space_charge.sc1d import addLongitudinalSpaceChargeNode
from orbit.space_charge.sc1d import SC1D_AccNode
from orbit.space_charge.sc2dslicebyslice.scLatticeModifications import setSC2DSliceBySliceAccNodes
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice_Coupled
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_Ring
from orbit.time_dep import TIME_DEP_Lattice
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

from SNS_RING import SNS_RING
from SNS_RING import X_FOIL
from SNS_RING import Y_FOIL

sys.path.append(os.getcwd())
from pyorbit_sim import utils


# Setup
# --------------------------------------------------------------------------------------

# MPI
_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)


# Create output directory.
outdir = "/home/46h/sim_data/"
path = pathlib.Path(__file__)
script_name = path.stem
outdir = os.path.join(
    outdir, 
    path.as_posix().split("scripts/")[1].split(".py")[0], 
    time.strftime("%Y-%m-%d"),
)
if _mpi_rank == 0:
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    print("Output directory: {}".format(outdir))
        
# Get timestamped output file prefix.
timestamp = ""
if _mpi_rank == 0:
    timestamp = time.strftime("%y%m%d%H%M%S")
timestamp = orbit_mpi.MPI_Bcast(timestamp, orbit_mpi.mpi_datatype.MPI_CHAR, 0, _mpi_comm)
prefix = "{}-{}".format(timestamp, script_name)
if _mpi_rank == 0:
    print(prefix)


def get_filename(filename):
    """Add output directory path and timestamp prefix to filename."""
    return os.path.join(outdir, "{}_{}".format(prefix, filename))


# Save a timestamped copy of this file. (It would be better to use version-control.)
shutil.copy(__file__, get_filename(".py"))

# Save git info
git_hash = utils.git_revision_hash()
git_url = utils.git_url()
if git_hash and git_url and utils.is_git_clean():
    print("Repository is clean.")
    print("Code should be available at")
    print("{}/-/tree/{}".format(git_url, git_hash))
else:
    print("Unknown code revision.")


# Simulation parameters
madx_file = "_input/SNS_RING_nux6.18_nuy6.18.lat"
madx_seq = "rnginj"
mass = mass_proton  # [GeV / c^2]
kin_energy = 0.600  # [GeV]
n_inj_turns = 501  # number of turns to inject
n_stored_turns = 0  # number of turns to store the beam after injection
kin_energy = 0.800  # synchronous particle energy [GeV]
mass = consts.mass_proton  # particle mass [GeV / c^2]
bunch_length_frac = 43.0 / 64.0  # macropulse length relative to ring length
n_macros_total = int(1.0e5)  # final number of macroparticles
n_macros_per_turn = int(n_macros_total / n_inj_turns)  # macroparticles per minipulse
nominal_n_inj_turns = 1000.0
nominal_intensity = 1.5e14
nominal_bunch_length_frac=(50.0 / 64.0)


info = open(get_filename("info.txt"), "w")
info.write("git_hash: {}\n".format(git_hash))
info.write("git_url: {}\n".format(git_url))


# # Lattice setup and linear analysis
# # --------------------------------------------------------------------------------------

# # Create time-dependent lattice.
# ring = SNS_RING()
# ring.readMADX(
#     "/home/46h/repo/accelerator-models/SNS/RING/SNS_RING_nux6.18_nuy6.18.lat",
#     "rnginj",
# )
# ring.initialize()


# # Linear lattice analysis (uncoupled)

# ## Turn off fringe fields. (This is affecting the MATRIX_Lattice tune calculation
# ## below. But shouldn't MATRIX_Lattice just extract the linear transfer matrix
# # from each node? TODO: Figure out what's going on.)
# ring.set_fringe_fields(False)

# ## Analyze the one-turn transfer matrix.
# test_bunch = Bunch()
# test_bunch.mass(mass)
# test_bunch.getSyncParticle().kinEnergy(kin_energy)
# matrix_lattice = TEAPOT_MATRIX_Lattice(ring, test_bunch)
# tmat_params = matrix_lattice.getRingParametersDict()
# if _mpi_rank == 0:
#     print("Transfer matrix parameters (uncoupled):")
#     pprint(tmat_params)

# ## Save parameters throughout the ring.
# (pos_nu_x, pos_alpha_x, pos_beta_x) = matrix_lattice.getRingTwissDataX()
# (pos_nu_y, pos_alpha_y, pos_beta_y) = matrix_lattice.getRingTwissDataY()
# twiss = pd.DataFrame()
# twiss["s"] = np.array(pos_nu_x)[:, 0]
# twiss["nu_x"] = np.array(pos_nu_x)[:, 1]
# twiss["nu_y"] = np.array(pos_nu_y)[:, 1]
# twiss["alpha_x"] = np.array(pos_alpha_x)[:, 1]
# twiss["alpha_y"] = np.array(pos_alpha_y)[:, 1]
# twiss["beta_x"] = np.array(pos_beta_x)[:, 1]
# twiss["beta_y"] = np.array(pos_beta_y)[:, 1]
# if _mpi_rank == 0:
#     twiss.to_csv(get_filename("lattice_twiss.dat"), sep=" ")

# (pos_disp_x, pos_disp_p_x) = matrix_lattice.getRingDispersionDataX()
# (pos_disp_y, pos_disp_p_y) = matrix_lattice.getRingDispersionDataY()
# dispersion = pd.DataFrame()
# dispersion["s"] = np.array(pos_disp_x)[:, 0]
# dispersion["disp_x"] = np.array(pos_disp_x)[:, 1]
# dispersion["disp_y"] = np.array(pos_disp_y)[:, 1]
# dispersion["disp_p_x"] = np.array(pos_disp_p_x)[:, 1]
# dispersion["disp_p_y"] = np.array(pos_disp_p_y)[:, 1]
# if _mpi_rank == 0:
#     dispersion.to_csv(get_filename("lattice_dispersion.dat"), sep=" ")


# # Linear lattice analysis (coupled)
            
# ## Analyze the one-turn transfer matrix.
# matrix_lattice = TEAPOT_MATRIX_Lattice_Coupled(ring, test_bunch, parameterization="LB")
# tmat_params = matrix_lattice.getRingParametersDict()
# if _mpi_rank == 0:
#     print("Transfer matrix parameters (coupled):")
#     pprint(tmat_params)
            
# ## Compute Twiss parameters throughout the ring.
# ## [...]

# ## Turn off fringe fields. (See previous comment.)
# ring.set_fringe_fields(True)


# # Bunch setup
# # --------------------------------------------------------------------------------------

# # Initialize the bunch.
# bunch = Bunch()
# bunch.mass(mass)
# sync_part = bunch.getSyncParticle()
# sync_part.kinEnergy(kin_energy)
# lostbunch = Bunch()
# lostbunch.addPartAttr("LostParticleAttributes")
# params_dict = {"bunch": bunch, "lostbunch": lostbunch}
# ring.set_bunch(bunch, lostbunch, params_dict)

# # Compute the macroparticle size from the minipulse width, intensity, and number
# # of macroparticles.
# nominal_minipulse_intensity = nominal_intensity / nominal_n_inj_turns
# nominal_bunch_length_frac = 50.0 / 64.0
# bunch_length_factor = bunch_length_frac / nominal_bunch_length_frac
# minipulse_intensity = nominal_minipulse_intensity * bunch_length_factor
# intensity = minipulse_intensity * n_inj_turns
# macro_size = intensity / n_inj_turns / n_macros_per_turn
# bunch.macroSize(macro_size)

            
# # Injection kicker optimization
# # --------------------------------------------------------------------------------------

# # Fringe fields complicate things. Turn them off for now.
# ring.set_fringe_fields(False)

# # Initial coordinates of closed orbit at injection point [x, x', y, y'].
# inj_coords_t0 = np.zeros(4)
# inj_coords_t0[0] = ring.x_foil
# inj_coords_t0[1] = 0.0
# inj_coords_t0[2] = ring.y_foil
# inj_coords_t0[3] = 0.0

# # Final coordinates of closed orbit at injection point  [x, x', y, y'].
# inj_coords_t1 = np.zeros(4)
# inj_coords_t1[0] = ring.x_foil - 0.035
# inj_coords_t1[1] = 0.0
# inj_coords_t1[2] = ring.y_foil
# inj_coords_t1[3] = -0.0015


# def plot_traj(trajectory, figname="fig.png"):
#     fig, ax = plt.subplots()
#     ax.plot(trajectory["s"], 1000.0 * trajectory["x"], marker=".")
#     ax.plot(trajectory["s"], 1000.0 * trajectory["y"], marker=".")
#     ax.set_ylim(-1.0, 65.0)
#     fig.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), figname))


# inj_controller = ring.get_injection_controller(
#     mass=mass, 
#     kin_energy=kin_energy, 
#     inj_mid="injm1", 
#     inj_start="bpm_a09", 
#     inj_end="bpm_b01",
# )

# if _mpi_rank == 0:
#     # (This is just stalling when I run with more than one MPI node; not sure why.)
    
#     # Remove kicker limits.
#     inj_controller.scale_kicker_limits(100.0)

#     # Bias the vertical orbit using vkickers.
#     inj_controller.set_inj_coords_vcorrectors([0.0, 0.0, 0.007, -0.0005], verbose=1)
#     inj_controller.print_inj_coords()
#     plot_traj(inj_controller.get_trajectory(), figname=get_filename("fig_inj_orbit_bias.png"))

#     # Set the initial phase space coordinates at the injection point.
#     solver_kws = dict(max_nfev=2500, verbose=2, ftol=1.0e-12, xtol=1.0e-12)
#     kicker_angles_t0 = inj_controller.set_inj_coords_fast(inj_coords_t0, **solver_kws)
#     inj_controller.print_inj_coords()
#     figname
#     plot_traj(inj_controller.get_trajectory(), figname=get_filename("fig_inj_orbit_t0.png"))

#     # Set the final phase space coordinates at the injection point.
#     kicker_angles_t1 = inj_controller.set_inj_coords_fast(inj_coords_t1, **solver_kws)
#     inj_controller.print_inj_coords()
#     plot_traj(inj_controller.get_trajectory(), figname=get_filename("fig_inj_orbit_t1.png"))

#     # Convert to list for MPI_Bcast
#     kicker_angles_t0 = kicker_angles_t0.tolist()
#     kicker_angles_t1 = kicker_angles_t1.tolist()

# # Get the kicker angles from rank 0.
# kicker_angles_t0 = orbit_mpi.MPI_Bcast(kicker_angles_t0, orbit_mpi.mpi_datatype.MPI_DOUBLE, 0, _mpi_comm)
# kicker_angles_t1 = orbit_mpi.MPI_Bcast(kicker_angles_t1, orbit_mpi.mpi_datatype.MPI_DOUBLE, 0, _mpi_comm)
# kicker_angles_t0 = np.array(kicker_angles_t0)
# kicker_angles_t1 = np.array(kicker_angles_t1)

# # Initialize the kickers.
# inj_controller.set_kicker_angles(kicker_angles_t0)

# # Assign sqrt(t) kicker waveforms.
# ring.setLatticeOrder()
# t0 = 0.0
# t1 = n_inj_turns * seconds_per_turn
# amps_t0 = np.ones(8)
# amps_t1 = np.abs(kicker_angles_t1 / kicker_angles_t0)
# for node, a0, a1 in zip(inj_controller.kicker_nodes, amps_t0, amps_t1):
#     waveform = SquareRootWaveform(t0=t0, t1=t1, a0=a0, a1=a1, sync_part=sync_part)
#     ring.setTimeDepNode(node.getParam("TPName"), waveform)

    
# # Injection
# # --------------------------------------------------------------------------------------

# # ring.add_inj_node()

# ring.add_foil_node(
#     xmin=-0.0085,
#     xmax=+0.0085,
#     ymin=-0.0080,
#     ymax=+0.1000,
#     thickness=390.0,
#     scatter="full",
# )


# # Apertures, collimators, and displacements
# # --------------------------------------------------------------------------------------

# ring.add_inj_chicane_aperture_displacement_nodes()
# ring.add_collimator_nodes()

# ## Aperture nodes need to be added as child nodes, but Jeff's benchmarks script selects
# ## the parent nodes by index; does not work with standard MADX output lattice.

# # ring.add_aperture_nodes()


# # RF cavities
# # --------------------------------------------------------------------------------------

# ring.add_rf_harmonic_nodes(
#     RF1=dict(phase=0.0, hnum=1.0, voltage=+2.0e-6),
#     RF2=dict(phase=0.0, hnum=2.0, voltage=-4.0e-6),
# )


# # Impedance
# # --------------------------------------------------------------------------------------

# ring.add_longitudinal_impedance_node(
#     n_macros_min=1000,
#     n_bins=128,
#     position=124.0,
#     ZL_Ekicker=None,  # read from file
#     ZL_RF=None,  # read from file
# )

# ring.add_transverse_impedance_node(
#     n_macros_min=1000,
#     n_bins=64,
#     use_x=0,
#     use_y=1,
#     position=124.0,
#     alpha_x=0.0,
#     alpha_y=-0.004,
#     beta_x=10.191,
#     beta_y=10.447,
#     q_x=6.21991,
#     q_y=6.20936,
# )


# # Space charge
# # --------------------------------------------------------------------------------------

# ring.add_longitudinal_space_charge_node(
#     b_a=(10.0 / 3.0),
#     n_macros_min=1000,
#     use=1,
#     n_bins=64,
#     position=124.0,
# )

# ring.add_transverse_space_charge_nodes(
#     n_macros_min=1000,
#     size_x=128,
#     size_y=128,
#     size_z=64,
#     path_length_min=1.0e-8,
#     n_boundary_points=128,
#     n_free_space_modes=32,
#     r_boundary=0.220,
#     kind="slicebyslice",
# )


# # Diagnostics
# # --------------------------------------------------------------------------------------

# # ring.add_tune_diagnostics_node(
# #     position=51.1921,
# #     beta_x=9.19025,
# #     alpha_x=-1.78574,
# #     eta_x=-0.000143012,
# #     etap_x=-2.26233e-05,
# #     beta_y=8.66549,
# #     alpha_y=0.538244,
# # )

# # ring.add_moments_diagnostics_node(order=4, position=0.0)



# # Tracking
# # --------------------------------------------------------------------------------------

# def should_dump_bunch(turn):
#     if turn % 100 == 0:
#         return True
#     if turn == n_inj_turns + n_stored_turns - 1:
#         return True
#     return False


# if _mpi_rank == 0:
#     print("Tracking.")
# for turn in trange(n_inj_turns + n_stored_turns):
#     ring.trackBunch(bunch, params_dict)
#     if should_dump_bunch(turn):
#         bunch.dumpBunch("_output/data/bunch_turn{}.dat".format(turn))


info.close()