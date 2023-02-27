# encoding=utf8  
from __future__ import print_function
import math
import os
import pathlib
import pickle
from pprint import pprint
import shutil
import sys
import time

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
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
from orbit.time_dep.waveform import ConstantWaveform
from orbit.time_dep.waveform import NthRootWaveform
from orbit.utils import consts
from orbit.utils.consts import mass_proton
from orbit.utils.consts import speed_of_light
import orbit_mpi
from spacecharge import LSpaceChargeCalc
from spacecharge import Boundary2D
from spacecharge import SpaceChargeCalc2p5D
from spacecharge import SpaceChargeCalcSliceBySlice2D

from SNS_RING import SNS_RING
from SNS_RING import X_INJ
from SNS_RING import Y_INJ

sys.path.append(os.getcwd())
from pyorbit_sim import utils
from pyorbit_sim.utils import ScriptManager


# Setup
# --------------------------------------------------------------------------------------

# MPI
_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

# Create output directory and save script info. (Not yet MPI compatible.)
man = ScriptManager(datadir="/home/46h/sim_data/", path=pathlib.Path(__file__))
man.save_info()
man.save_script_copy()
print("Script info:")
pprint(man.get_info())

# Run parameters
madx_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 
    "data/SNS_RING_nux6.18_nuy6.18_dual_solenoid/LATTICE.lat",
)
madx_seq = "rnginjsol"
mass = mass_proton  # particle mass [GeV / c^2]
kin_energy = 0.800  # synchronous particle energy [GeV]
n_inj_turns = 1  # number of injected turns
n_turns_track = 100  # number of turns to simulate
dump_bunch_every = 1  # dump the bunch coordinates after this many turns
n_macros_total = int(1.0e4)  # final number of macroparticles
n_macros_per_turn = int(n_macros_total / n_inj_turns)  # macroparticles per minipulse
bunch_length_frac = 64.0 / 64.0  # bunch length as fraction of ring

switches = {
    "apertures": False,
    "impedance_transverse": False,
    "impedance_longitudinal": False,
    "foil": False,
    "fringe": False,
    "RF": False,
    "solenoid": False,
    "space_charge_transverse": False,
    "space_charge_longitudinal": False,
}


# Lattice setup and linear analysis
# --------------------------------------------------------------------------------------

# Create time-dependent lattice.
ring = SNS_RING()
ring.readMADX(madx_file, madx_seq)
ring.initialize()

# Toggle solenoid.
for name in ["scbdsol_c13a", "scbdsol_c13b"]:
    node = ring.getNodeForName(name)
    B =  0.6 / (2.0 * node.getLength())
    if switches["solenoid"]:
        B = 0.6 / (2.0 * node.getLength())
    else:
        B = 0.0
    node.setParam("B", B)
    print("{}: B={:.2f}, L={:.2f}".format(node.getName(), node.getParam("B"), node.getLength()))

    
# Linear lattice analysis (uncoupled)

## Turn off fringe fields.
ring.set_fringe_fields(False)

## Analyze the one-turn transfer matrix.
test_bunch = Bunch()
test_bunch.mass(mass)
test_bunch.getSyncParticle().kinEnergy(kin_energy)
matrix_lattice = TEAPOT_MATRIX_Lattice(ring, test_bunch)
tmat_params = matrix_lattice.getRingParametersDict()
if _mpi_rank == 0:
    print("Transfer matrix parameters (uncoupled):")
    pprint(tmat_params)
file = open(man.get_filename("lattice_params_uncoupled.pkl"), "wb")
pickle.dump(tmat_params, file, protocol=pickle.HIGHEST_PROTOCOL)
file.close()

## Save parameters throughout the ring.
twiss = ring.get_ring_twiss(mass=mass, kin_energy=kin_energy)
dispersion = ring.get_ring_dispersion(mass=mass, kin_energy=kin_energy)
if _mpi_rank == 0:
    print("Twiss:")
    print(twiss)
    twiss.to_csv(man.get_filename("lattice_twiss.dat"), sep=" ")
    print("Dispersion:")
    print(dispersion)
    dispersion.to_csv(man.get_filename("lattice_dispersion.dat"), sep=" ")


# Linear lattice analysis (coupled)
            
## Analyze the one-turn transfer matrix.
matrix_lattice = TEAPOT_MATRIX_Lattice_Coupled(ring, test_bunch, parameterization="LB")
tmat_params = matrix_lattice.getRingParametersDict()
if _mpi_rank == 0:
    print("Transfer matrix parameters (coupled):")
    pprint(tmat_params)
file = open(man.get_filename("lattice_params_coupled.pkl"), "wb")
pickle.dump(tmat_params, file, protocol=pickle.HIGHEST_PROTOCOL)
file.close()

## Compute Twiss parameters throughout the ring.
twiss_coupled = ring.get_ring_twiss_coupled(mass=mass, kin_energy=kin_energy)
if _mpi_rank == 0:
    print("Twiss:")
    print(twiss_coupled)
    twiss.to_csv(man.get_filename("lattice_twiss_coupled.dat"), sep=" ")

## Turn on fringe fields.
ring.set_fringe_fields(switches["fringe"])


# Bunch setup
# --------------------------------------------------------------------------------------

# Initialize the bunch.
bunch = Bunch()
bunch.mass(mass)
sync_part = bunch.getSyncParticle()
sync_part.kinEnergy(kin_energy)
lostbunch = Bunch()
lostbunch.addPartAttr("LostParticleAttributes")
params_dict = {"bunch": bunch, "lostbunch": lostbunch}
ring.set_bunch(bunch, lostbunch, params_dict)

# Compute the macroparticle size from the minipulse width, intensity, and number
# of macroparticles.
nominal_n_inj_turns = 1000.0
nominal_intensity = 1.5e14
nominal_minipulse_intensity = nominal_intensity / nominal_n_inj_turns
nominal_bunch_length_frac = 50.0 / 64.0
bunch_length_factor = bunch_length_frac / nominal_bunch_length_frac
minipulse_intensity = nominal_minipulse_intensity * bunch_length_factor
intensity = minipulse_intensity * n_inj_turns
macro_size = intensity / n_inj_turns / n_macros_per_turn
bunch.macroSize(macro_size)

            
# Injection kicker optimization
# --------------------------------------------------------------------------------------

# Define center of injected distribution.
x_inj = X_INJ
y_inj = Y_INJ

# Initial coordinates of closed orbit at injection point [x, x', y, y'].
inj_coords_t0 = np.zeros(4)
inj_coords_t0[0] = x_inj
inj_coords_t0[1] = 0.0
inj_coords_t0[2] = y_inj
inj_coords_t0[3] = 0.0

# Final coordinates of closed orbit at injection point  [x, x', y, y'].
inj_coords_t1 = np.zeros(4)
inj_coords_t1[0] = x_inj- 0.025
inj_coords_t1[1] = 0.000
inj_coords_t1[2] = y_inj - 0.25
inj_coords_t1[3] = 0.000

# Fringe fields complicate things. Turn them off for now.
ring.set_fringe_fields(False)


inj_controller = ring.get_injection_controller(
    mass=mass, 
    kin_energy=kin_energy, 
    inj_mid="injm1", 
    inj_start="bpm_a09", 
    inj_end="bpm_b01",
)
inj_controller.scale_kicker_limits(100.0)
solver_kws = dict(max_nfev=2500, verbose=1, ftol=1.0e-12, xtol=1.0e-12)

## Bias the vertical orbit using the vkickers.
bias = False
if bias:
    print("Biasing vertical orbit using vkickers")
    inj_controller.set_inj_coords_vcorrectors([0.0, 0.0, 0.007, -0.0005], verbose=1)
    inj_controller.print_inj_coords()
traj = inj_controller.get_trajectory()
traj.to_csv(man.get_filename("inj_orbit_bias.dat"), sep=" ")

## Set the initial phase space coordinates at the injection point.
print("Optimizing kickers (t=0)")
kicker_angles_t0 = inj_controller.set_inj_coords_fast(inj_coords_t0, **solver_kws)
inj_controller.print_inj_coords()
traj = inj_controller.get_trajectory()
traj.to_csv(man.get_filename("inj_orbit_t0.dat"), sep=" ")

## Set the final phase space coordinates at the injection point.
print("Optimizing kickers (t=1)")
kicker_angles_t1 = inj_controller.set_inj_coords_fast(inj_coords_t1, **solver_kws)
inj_controller.print_inj_coords()
traj = inj_controller.get_trajectory()
traj.to_csv(man.get_filename("inj_orbit_t1.dat"), sep=" ")

# Set kicker waveforms.
seconds_per_turn = ring.getLength() / (sync_part.beta() * consts.speed_of_light)
t0 = 0.0
t1 = n_inj_turns * seconds_per_turn
strengths_t0 = np.ones(8)
strengths_t1 = np.abs(kicker_angles_t1 / kicker_angles_t0)
for node, s0, s1 in zip(inj_controller.kicker_nodes, strengths_t0, strengths_t1):
    waveform = NthRootWaveform(n=2.0, t0=t0, t1=t1, s0=s0, s1=s1, sync_part=sync_part)
    # node.setWaveform(waveform)

# Set kickers to t0 state.
inj_controller.set_kicker_angles(kicker_angles_t0)

# Toggle fringe fields (see previous above).
ring.set_fringe_fields(switches["fringe"])

    
# Injection
# --------------------------------------------------------------------------------------

# Set absolute limits on injected coordinates (foil boundaries)
xmin = x_inj - 0.0085
xmax = x_inj + 0.0085
ymin = y_inj - 0.0080
ymax = y_inj + 0.1000

# Add injection node at lattice entrance.
ring.add_inj_node(
    n_parts=n_macros_per_turn,
    n_parts_max=n_macros_total,
    xmin=xmin,
    xmax=xmax,
    ymin=ymin,
    ymax=ymax,
    dist_x_kind="joho",
    dist_y_kind="joho",
    dist_z_kind="uniform",  # {"snsespread", "uniform"}
    dist_x_kws={
        "centerpos": x_inj,
        "centermom": 0.0,
        "eps_rms": 0.221e-6,
    },
    dist_y_kws={
        "centerpos": y_inj,
        "centermom": 0.0,
        "eps_rms": 0.221e-6,
    },
    dist_z_kws={
        "n_inj_turns": n_inj_turns,
        "bunch_length_frac": bunch_length_frac,
    }
)

if switches["foil"]:
    ring.add_foil_node(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        thickness=390.0,
        scatter="full",
    )


# Apertures, collimators, and displacements
# --------------------------------------------------------------------------------------

if switches["apertures"]:
    ring.add_inj_chicane_aperture_displacement_nodes()
    ring.add_collimator_nodes()
    # ring.add_aperture_nodes()


# RF cavities
# --------------------------------------------------------------------------------------

if switches["RF"]:
    ring.add_rf_harmonic_nodes(
        RF1=dict(phase=0.0, hnum=1.0, voltage=+2.0e-6),
        RF2=dict(phase=0.0, hnum=2.0, voltage=-4.0e-6),
    )


# Impedance
# --------------------------------------------------------------------------------------

if switches["impedance_longitudinal"]:
    ring.add_longitudinal_impedance_node(
        n_macros_min=1000,
        n_bins=128,
        position=124.0,
        ZL_Ekicker=None,  # read from file
        ZL_RF=None,  # read from file
    )

if switches["impedance_transverse"]:
    ring.add_transverse_impedance_node(
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
    )


# Space charge
# --------------------------------------------------------------------------------------

if switches["space_charge_longitudinal"]:
    ring.add_longitudinal_space_charge_node(
        b_a=(10.0 / 3.0),
        n_macros_min=1000,
        use=1,
        n_bins=64,
        position=124.0,
    )

if switches["space_charge_transverse"]:
    ring.add_transverse_space_charge_nodes(
        n_macros_min=1000,
        size_x=128,
        size_y=128,
        size_z=64,
        path_length_min=1.0e-8,
        n_boundary_points=128,
        n_free_space_modes=32,
        r_boundary=0.220,
        kind="slicebyslice",
    )


# Diagnostics
# --------------------------------------------------------------------------------------

# There is not much dispersion at the injection point...
index = 0
ring.add_tune_diagnostics_node(
    filename=man.get_filename("tunes.dat"),
    position=twiss.loc[index, "s"],
    beta_x=twiss.loc[index, "beta_x"],
    beta_y=twiss.loc[index, "beta_y"],
    alpha_x=twiss.loc[index, "alpha_x"],
    alpha_y=twiss.loc[index, "alpha_y"],
    eta_x=dispersion.loc[index, "disp_x"],
    etap_x=dispersion.loc[index, "dispp_x"],
)

## Second-order moments node
# [...]


# Tracking
# --------------------------------------------------------------------------------------

def should_dump_bunch(turn):
    if turn == 1 or turn == n_turns_track:
        return True
    if turn % dump_bunch_every == 0:
        return True
    return False


if _mpi_rank == 0:
    print("Tracking.")
    
# Flat-top the kickers
inj_controller.set_kicker_angles(kicker_angles_t1)
        
for turn in trange(1, n_turns_track + 1):
    ring.trackBunch(bunch, params_dict)    
    if should_dump_bunch(turn):
        filename = man.get_filename("bunch_turn{}.dat".format(turn))
        bunch.dumpBunch(filename)
        
if _mpi_rank == 0:
    print("Script info:")
    pprint(man.get_info())
