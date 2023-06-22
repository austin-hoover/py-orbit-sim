"""SNS linac simulation."""
from __future__ import print_function
import math
import os
import pathlib
import pickle
from pprint import pprint
import random
import shutil
import sys
import time

import numpy as np

from bunch import Bunch
from bunch import BunchTwissAnalysis
from bunch_utils_functions import copyCoordsToInitCoordsAttr
from linac import BaseRfGap
from linac import BaseRfGap_slow
from linac import MatrixRfGap
from linac import RfGapThreePointTTF
from linac import RfGapThreePointTTF_slow
from linac import RfGapTTF
from linac import RfGapTTF_slow
from orbit.bunch_generators import GaussDist3D
from orbit.bunch_generators import KVDist3D
from orbit.bunch_generators import TwissContainer
from orbit.bunch_generators import WaterBagDist3D
from orbit.bunch_utils import ParticleIdNumber
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.py_linac.lattice import AxisFieldRF_Gap
from orbit.py_linac.lattice import AxisField_and_Quad_RF_Gap
from orbit.py_linac.lattice import BaseRF_Gap
from orbit.py_linac.lattice import Bend
from orbit.py_linac.lattice import Drift
from orbit.py_linac.lattice import LinacApertureNode
from orbit.py_linac.lattice import LinacEnergyApertureNode
from orbit.py_linac.lattice import LinacPhaseApertureNode
from orbit.py_linac.lattice import OverlappingQuadsNode 
from orbit.py_linac.lattice import Quad
from orbit.py_linac.lattice_modifications import Add_quad_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_rfgap_apertures_to_lattice
from orbit.py_linac.lattice_modifications import AddMEBTChopperPlatesAperturesToSNS_Lattice
from orbit.py_linac.lattice_modifications import AddScrapersAperturesToLattice
from orbit.py_linac.lattice_modifications import GetLostDistributionArr
from orbit.py_linac.lattice_modifications import Replace_BaseRF_Gap_and_Quads_to_Overlapping_Nodes
from orbit.py_linac.lattice_modifications import Replace_BaseRF_Gap_to_AxisField_Nodes
from orbit.py_linac.lattice_modifications import Replace_Quads_to_OverlappingQuads_Nodes
from orbit.py_linac.linac_parsers import SNS_LinacLatticeFactory
from orbit.py_linac.overlapping_fields import SNS_EngeFunctionFactory
from orbit.space_charge.sc3d import setSC3DAccNodes
from orbit.space_charge.sc3d import setUniformEllipsesSCAccNodes
from orbit.space_charge.sc2p5d import setSC2p5DrbAccNodes
from orbit.utils import consts
import orbit_mpi
from spacecharge import SpaceChargeCalc3D
from spacecharge import SpaceChargeCalcUnifEllipse

# Local
from SNS_LINAC import SNS_LINAC

import pyorbit_sim.bunch_utils
import pyorbit_sim.linac
import pyorbit_sim.plotting
from pyorbit_sim.misc import lorentz_factors
from pyorbit_sim.utils import ScriptManager


# Setup
# --------------------------------------------------------------------------------------

settings = {
    "output": {
        "datadir": "./scripts/SNS_LINAC/data_output",  # output directory location
        "save": True,  # no output if False
    },
    "lattice": {
        "rf_frequency": 402.5e+06,  # [Hz]
        "max_drift_length": 0.010,  # [m]
        "sequences": [
            "MEBT",
            "DTL1",
            "DTL2",
            "DTL3",
            "DTL4",
            "DTL5",
            # "DTL6",
            # "CCL1",
            # "CCL2",
            # "CCL3",
            # "CCL4",
            # "SCLMed",
            # "SCLHigh",
            # "HEBT1",
            # "HEBT2",
        ],
        "rf_gap_model": RfGapTTF,
        "overlapping_fields": {
            "switch": True,
            "sequences": "all",
            "z_step": 0.002, 
            "xml_filename": "sns_rf_fields.xml",
        },
        "linac_tracker": True,
        "space_charge": {
            "switch": True,
            "solver": "FFT",
            "grid_size": (64, 64, 64),
            "n_ellipsoids": 5,
            "path_length_min": 0.010,  # [m]
        },
        "apertures": {
            "transverse": {
                "switch": True,
                "x_max": 0.042,  # [m]
                "y_max": 0.042,  # [m]
            },
            "longitudinal": {
                "switch": True,
                "phase_min": -180.0, 
                "phase_max": +180.0, 
                "energy_min": -0.100,
                "energy_max": +0.100,
            },
        },
    },
    "bunch": {
        "filename": None,
        "rms_equiv": None,
        "mass": 0.939294,  # [GeV / c^2]
        "charge": -1.0,  # [elementary charge units]
        "kin_energy": 0.0025,  # [GeV]
        "current":  0.042,  # [A]
        "design": {
            "dist": WaterBagDist3D,
            "alpha_x": -1.9620,
            "alpha_y": 1.7681,
            "alpha_z": -0.0196,
            "beta_x": 0.1831,
            "beta_y": 0.1620,
            "beta_z": 0.5844,
            "eps_x": 0.21e-06,
            "eps_y": 0.21e-06,
            "eps_z": 0.24153e-06,
        },   
    },
    "sim": {
    },
}

# MPI
_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

# Broadcast timestamp to all nodes from MPI rank 0.
main_rank = 0
datestamp = time.strftime("%Y-%m-%d")
timestamp = time.strftime("%y%m%d%H%M%S")
datestamp = orbit_mpi.MPI_Bcast(datestamp, orbit_mpi.mpi_datatype.MPI_CHAR, main_rank, _mpi_comm)
timestamp = orbit_mpi.MPI_Bcast(timestamp, orbit_mpi.mpi_datatype.MPI_CHAR, main_rank, _mpi_comm)

# Create output directory and save script info.
man = ScriptManager(
    datadir=settings["output"]["datadir"],
    path=pathlib.Path(__file__), 
    timestamp=timestamp,
    datestamp=datestamp,
)
if _mpi_rank == 0 and settings["output"]["save"]:
    man.make_outdir()
    man.save_info()
    man.save_script_copy()
    pprint(man.get_info())
file_path = os.path.dirname(os.path.realpath(__file__))

# Derive some settings.
settings["output"].update(**man.get_info())
if settings["lattice"]["overlapping_fields"]["sequences"] == "all":
    settings["lattice"]["overlapping_fields"]["sequences"] = settings["lattice"]["sequences"]
settings["bunch"]["intensity"] = pyorbit_sim.bunch_utils.get_intensity(
    settings["bunch"]["current"], 
    settings["lattice"]["rf_frequency"]
)

# Print settings.
if _mpi_rank == 0:
    print("Settings:")
    pprint(settings)


# Lattice 
# --------------------------------------------------------------------------------------

linac = SNS_LINAC(
    input_dir=os.path.join(file_path, "data_input"),
    xml_filename="sns_linac.xml",
    rf_frequency=settings["lattice"]["rf_frequency"],
)

lattice = linac.initialize(
    sequences=settings["lattice"]["sequences"], 
    max_drift_length=settings["lattice"]["max_drift_length"],
    verbose=True,
)

linac.save_node_positions(man.get_filename("lattice_nodes.txt"))
linac.save_lattice_structure(man.get_filename("lattice_structure.txt"))

linac.set_rf_gap_model(settings["lattice"]["rf_gap_model"])

if settings["lattice"]["overlapping_fields"]["switch"]:
    linac.set_overlapping_rf_and_quad_fields(
        sequences=settings["lattice"]["overlapping_fields"]["sequences"],
        z_step=settings["lattice"]["overlapping_fields"]["z_step"],
        xml_filename=settings["lattice"]["overlapping_fields"]["xml_filename"]
    )
    
linac.set_linac_tracker(settings["lattice"]["linac_tracker"])

if settings["lattice"]["space_charge"]["switch"]:
    linac.add_space_charge_nodes(
        solver=settings["lattice"]["space_charge"]["solver"],
        grid_size=settings["lattice"]["space_charge"]["grid_size"],
        n_ellipsoids=settings["lattice"]["space_charge"]["n_ellipsoids"], 
        path_length_min=settings["lattice"]["space_charge"]["path_length_min"],
        verbose=True,
    )

if settings["lattice"]["apertures"]["transverse"]["switch"]:
    linac.add_transverse_aperture_nodes(
        x_size=settings["lattice"]["apertures"]["transverse"]["x_max"],
        y_size=settings["lattice"]["apertures"]["transverse"]["y_max"],
        verbose=True
    )
if settings["lattice"]["apertures"]["transverse"]["switch"]:
    linac.add_longitudinal_apertures(
        classes=[
            BaseRF_Gap, 
            AxisFieldRF_Gap, 
            AxisField_and_Quad_RF_Gap,
            Quad, 
            OverlappingQuadsNode,
        ],
        phase_min=settings["lattice"]["apertures"]["longitudinal"]["phase_min"],
        phase_max=settings["lattice"]["apertures"]["longitudinal"]["phase_max"],
        energy_min=settings["lattice"]["apertures"]["longitudinal"]["energy_min"],
        energy_max=settings["lattice"]["apertures"]["longitudinal"]["energy_max"],
        verbose=True,
    )
    
lattice = linac.lattice
aperture_nodes = linac.aperture_nodes


# Bunch
# --------------------------------------------------------------------------------------

bunch_filename = os.path.join(
    "/home/46h/projects/BTF/sim/SNS_LINAC/2023-06-18_RFQ-WS04b_PARMTEQ/data/derived/",
    "230618191218-bunch_MEBT_Diag:WS04b_upsample_1.00e+08_decorr_x-y-z.dat"
)
rms_equiv_dist = None


# # Initialize the bunch.
# bunch = Bunch()
# bunch.mass(settings["bunch"]["mass"])
# bunch.charge(settings["bunch"]["charge"])
# bunch.getSyncParticle().kinEnergy(settings["bunch"]["kin_energy"])



# intensity = (settings["bunch"]["current"] / settings["lattice"]["rf_frequency"]) / abs(float(bunch.charge()) * consts.charge_electron)

# # Load the bunch coordinates.
# if bunch_filename is not None:
#     if _mpi_rank == 0:
#         print("Generating bunch from file '{}'.".format(bunch_filename))
#     bunch.readBunch(bunch_filename)
# else:
#     dist = GaussDist3D
#     n_parts = int(1e4)
#     kin_energy = 0.0025  # [GeV]
#     mass = 0.939294  # [GeV / c^2]
#     gamma = (mass + kin_energy) / mass
#     beta = math.sqrt(gamma * gamma - 1.0) / gamma
#     alpha_x = -1.9620
#     alpha_y = 1.7681
#     alpha_z = -0.0196
#     beta_x = 0.1831
#     beta_y = 0.1620
#     beta_z = 0.5844
#     eps_x = 0.21e-6 / (beta * gamma)  # [m * rad]
#     eps_y = 0.21e-6 / (beta * gamma)  # [m * rad]
#     eps_z = 0.24153e-6 / (beta * gamma**3)  # [m * rad]
#     eps_z = eps_z * gamma**3 * beta**2 * bunch.mass()  # [m * GeV]
#     beta_z = beta_z / (gamma**3 * beta**2 * bunch.mass())    
#     if _mpi_rank == 0:
#         print("Generating bunch from design Twiss parameters and {} generator.".format(dist))    
#     bunch = pyorbit_sim.bunch_utils.generate(
#         dist=dist(
#             twissX=TwissContainer(alpha_x, beta_x, eps_x),
#             twissY=TwissContainer(alpha_y, beta_y, eps_y),
#             twissZ=TwissContainer(alpha_z, beta_z, eps_z),
#         ),
#         bunch=bunch,
#         n_parts=n_parts,
#         verbose=True,
#     )
    
# # Set bunch centroid to zero. 
# pyorbit_sim.bunch_utils.center(bunch, verbose=True)
        
# # Generate an RMS-equivalent distribution in x-x', y-y', and z-z' using an analytic 
# # distribution function.
# if False:
#     dist = GaussDist3D
#     n_parts = bunch.getSizeGlobal()
#     if _mpi_rank == 0:
#         print("Forming rms-equivalent bunch using 2D Twiss parameters and {} generator.".format(dist))
#     bunch_twiss_analysis = BunchTwissAnalysis()
#     order = 2
#     dispersion_flag = 0
#     emit_norm_flag = 0
#     bunch_twiss_analysis.computeBunchMoments(bunch, order, dispersion_flag, emit_norm_flag)    
#     eps_x = bunch_twiss_analysis.getEffectiveEmittance(0)
#     eps_y = bunch_twiss_analysis.getEffectiveEmittance(1)
#     eps_z = bunch_twiss_analysis.getEffectiveEmittance(2)
#     beta_x = bunch_twiss_analysis.getEffectiveBeta(0)
#     beta_y = bunch_twiss_analysis.getEffectiveBeta(1)
#     beta_z = bunch_twiss_analysis.getEffectiveBeta(2)
#     alpha_x = bunch_twiss_analysis.getEffectiveAlpha(0)
#     alpha_y = bunch_twiss_analysis.getEffectiveAlpha(1)
#     alpha_z = bunch_twiss_analysis.getEffectiveAlpha(2)    
#     bunch = pyorbit_sim.bunch_utils.generate(
#         dist=dist(
#             twissX=TwissContainer(alpha_x, beta_x, eps_x),
#             twissY=TwissContainer(alpha_y, beta_y, eps_y),
#             twissZ=TwissContainer(alpha_z, beta_z, eps_z),
#         ),
#         n_parts=n_parts, 
#         bunch=bunch, 
#         verbose=True,
#     )
    
# # Downsample. (Assume the particles were randomly generated to begin with.)
# fraction_keep = None
# if fraction_keep and fraction_keep < 1.0:
#     n = int(fraction_keep * bunch.getSize())  # on each processor
#     print("(rank {}) Downsampling by factor {}.".format(_mpi_rank, 1.0 / fraction_keep))
#     for i in reversed(range(n, bunch.getSize())):
#         bunch.deleteParticleFast(i)
#     bunch.compress()    
        
# # Decorrelate x-y-z. (Not working with MPI).
# if False:
#     bunch = pyorbit_sim.bunch_utils.decorrelate_x_y_z(bunch, verbose=True)

    
# # Set the macro-particle size.
# bunch_size_global = bunch.getSizeGlobal()
# macro_size = intensity / bunch_size_global
# bunch.macroSize(macro_size)
    
# # Print bunch parameters.
# twiss_analysis = BunchTwissAnalysis()
# twiss_analysis.analyzeBunch(bunch)
# (alpha_x, beta_x, _, eps_x) = twiss_analysis.getTwiss(0)
# (alpha_y, beta_y, _, eps_y) = twiss_analysis.getTwiss(1)
# (alpha_z, beta_z, _, eps_z) = twiss_analysis.getTwiss(2)
# if _mpi_rank == 0:
#     print("Bunch parameters:")
#     print("  charge = {}".format(bunch.charge()))
#     print("  mass = {} [GeV / c^2]".format(bunch.mass()))
#     print("  kinetic energy = {} [GeV]".format(bunch.getSyncParticle().kinEnergy()))
#     print("  macrosize = {}".format(bunch.macroSize()))
#     print("  size (local) = {:.2e}".format(bunch.getSize()))
#     print("  size (global) = {:.2e}".format(bunch_size_global))    
#     print("Twiss parameters:")
#     print("  alpha_x = {}".format(alpha_x))
#     print("  alpha_y = {}".format(alpha_y))
#     print("  alpha_z = {}".format(alpha_z))
#     print("  beta_x = {}".format(beta_x))
#     print("  beta_y = {}".format(beta_y))
#     print("  beta_z = {}".format(beta_z))
#     print("  eps_x = {} [mm * mrad]".format(1.0e6 * eps_x))
#     print("  eps_y = {} [mm * mrad]".format(1.0e6 * eps_y))
#     print("  eps_z = {} [mm * MeV]".format(1.0e6 * eps_z))
# if _mpi_rank == 0:
#     print("Centroid coordinates:")
# dims = ["x", "xp", "y", "yp", "z", "dE"]
# units = ["m", "rad", "m", "rad", "m", "GeV"]
# for i, (dim, unit) in enumerate(zip(dims, units)):
#     mean = twiss_analysis.getAverage(i)
#     if _mpi_rank == 0:
#         print("  <{}> = {:.3e} [{}]".format(dim, mean, unit))
    
    
# ## Assign ID number to each particle.
# # ParticleIdNumber.addParticleIdNumbers(bunch)
# # copyCoordsToInitCoordsAttr(bunch)



# # Tracking
# # --------------------------------------------------------------------------------------

# start = 0.0  # start node (name/position/None)
# stop = None  # stop node (name/position/None)
# save_input_bunch = True
# save_output_bunch = True
# stride = {
#     "update": 0.100,  # [m]
#     "write_bunch": 20.0,  # [m]
#     "plot_bunch": np.inf,  # [m]
# }


# # Create bunch writer.
# writer = pyorbit_sim.linac.BunchWriter(
#     folder=man.outdir, 
#     prefix=man.prefix, 
#     index=0,
# )    
    
    
# # Create bunch plotter. (Does not currently work with MPI.)
# def transform(X):
#     X[:, :] *= 1000.0
#     X = pyorbit_sim.bunch_utils.norm_xxp_yyp_zzp(X, scale_emittance=True)
#     X = pyorbit_sim.bunch_utils.slice_sphere(
#         X,
#         axis=(0, 1, 2, 3),
#         rmin=0.0,
#         rmax=1.0,
#     )
#     return X

# plotter = pyorbit_sim.plotting.Plotter(
#     transform=transform, 
#     folder=man.outdir,
#     prefix=man.prefix,
#     default_save_kws=None, 
#     dims=["x", "xp", "y", "yp", "z", "dE"],
# )
# plot_kws = dict(
#     text=True,
#     colorbar=True,
#     floor=1.0,
#     divide_by_max=True,
#     profx=True,
#     profy=True,
#     bins=75,
#     norm="log",
# )
# plotter.add_function(
#     pyorbit_sim.plotting.proj2d, 
#     axis=(4, 5),
#     limits=[(-5.0, 5.0), (-5.0, 5.0)],
#     save_kws=dict(dpi=200), 
#     name=None, 
#     **plot_kws
# )


# # Ignore writer and plotter if not saving output.
# if not save:
#     writer = None
#     plotter = None

    
# # Create bunch monitor.
# monitor = pyorbit_sim.linac.Monitor(
#     position_offset=0.0,  # will be set automatically in `pyorbit_sim.linac.track`.
#     stride=stride,
#     writer=writer,
#     plotter=plotter,
#     track_history=True,
#     track_rms=True,
#     dispersion_flag=False,
#     emit_norm_flag=False,
#     rf_frequency=rf_frequency,
#     verbose=True,
#     filename=man.get_filename("history.dat"),  # saves every update step
# )


# # Record synchronous particle time of arrival at each accelerating cavity.
# if _mpi_rank == 0:
#     print("Tracking design bunch...")
# design_bunch = lattice.trackDesignBunch(bunch)
# if _mpi_rank == 0:
#     print("Design bunch tracking complete.")
    
    
# # Check the synchronous particle time. This could be wrong if start != 0 and
# # if the bunch was loaded from a file without a PyORBIT header.
# pyorbit_sim.linac.check_sync_part_time(bunch, lattice, start=start, set_design=True)
    
    
# # Save input bunch.
# if save and save_input_bunch:
#     node_name = start
#     if node_name is None or type(node_name) is not str:
#         node_name = "START"
#     writer.action(bunch, node_name)
    
    
# # Track
# if _mpi_rank == 0:
#     print("Tracking...")
# params_dict = pyorbit_sim.linac.track(
#     bunch, 
#     lattice, 
#     monitor=monitor, 
#     start=start, 
#     stop=stop, 
#     verbose=True
# )
    
    
# # Save losses vs. position.
# aprt_nodes_losses = GetLostDistributionArr(aperture_nodes, params_dict["lostbunch"])
# if _mpi_rank == 0:
#     print("Total loss = {:.2e}".format(sum([loss for (node, loss) in aprt_nodes_losses])))
# if save:
#     filename = man.get_filename("losses.txt")
#     if _mpi_rank == 0:
#         print("Saving loss vs. node array to {}".format(filename))
#     file = open(filename, "w")
#     file.write("node position loss\n")
#     for (node, loss) in aprt_nodes_losses:
#         file.write("{} {} {}\n".format(node.getName(), node.getPosition(), loss))
#     file.close()
    
    
# # Save output bunch.
# if save and save_output_bunch:
#     node_name = stop 
#     if node_name is None or type(node_name) is not str:
#         node_name = "STOP"
#     writer.action(bunch, node_name)

    
# if _mpi_rank == 0:
#     print("timestamp = {}".format(man.timestamp))
