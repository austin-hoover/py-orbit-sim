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
        "save": False,  # no output if False
    },
    "lattice": {
        "apertures": {
            "transverse": {
                "switch": True,
                "x_max": 0.042,  # [m]
                "y_max": 0.042,  # [m]
            },
            "longitudinal": {
                "switch": True,
                "energy_min": -0.100,
                "energy_max": +0.100,
                "phase_min": -180.0, 
                "phase_max": +180.0, 
            },
        },
        "linac_tracker": True,
        "max_drift_length": 0.010,  # [m]
        "overlapping_fields": {
            "switch": True,
            "sequences": "all",
            "xml_filename": "sns_rf_fields.xml",
            "z_step": 0.002, 
        },
        "rf_frequency": 402.5e+06,  # [Hz]
        "rf_gap_model": RfGapTTF,
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
        "space_charge": {
            "switch": True,
            "solver": "FFT",
            "grid_size": (64, 64, 64),
            "n_ellipsoids": 5,
            "path_length_min": 0.010,  # [m]
        },
    },
    "bunch": {
        "filename": None, 
        "rms_equiv": None,
        "mass": 0.939294,  # [GeV / c^2]
        "charge": -1.0,  # [elementary charge units]
        "kin_energy": 0.0025,  # [GeV]
        "current":  0.042,  # [A]
        "centroid": 0.0,
        "downsample": None,
        "decorrelate_x-y-z": False,
        "design": {
            "dist": WaterBagDist3D,
            "n_parts": 10000,
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
        "start": 0.0,  # (node name/position/None)
        "stop": None,  # (node name/position/None)
        "save_input_bunch": True,
        "save_output_bunch": True,
        "particle_ids": False,
        "stride": {
            "update": 0.100,  # [m]
            "write_bunch": 20.0,  # [m]
            "plot_bunch": np.inf,  # [m]
        },
        "set_design_sync_time": True,
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

if settings["output"]["save"]:
    linac.save_node_positions(man.get_filename("lattice_nodes.txt"))
    linac.save_lattice_structure(man.get_filename("lattice_structure.txt"))

linac.set_rf_gap_model(settings["lattice"]["rf_gap_model"])

if settings["lattice"]["overlapping_fields"]["switch"]:
    sequences = settings["lattice"]["sequences"]
    if sequences == "all":
        sequences = settings["lattice"]["sequences"]
    linac.set_overlapping_rf_and_quad_fields(
        sequences=sequences,
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

# Initialize the bunch.
bunch = Bunch()
bunch.mass(settings["bunch"]["mass"])
bunch.charge(settings["bunch"]["charge"])
bunch.getSyncParticle().kinEnergy(settings["bunch"]["kin_energy"])

# Load the bunch coordinates. If no filename is provided, generate a design bunch. 
if settings["bunch"]["filename"] is not None:
    bunch.readBunch(bunch_filename, verbose=True)
else:
    bunch = pyorbit_sim.bunch_utils.generate_from_norm_twiss(
        dist=settings["bunch"]["design"]["dist"],
        n_parts=settings["bunch"]["design"]["n_parts"],
        bunch=bunch,
        verbose=True,
        alpha_x=settings["bunch"]["design"]["alpha_x"],
        alpha_y=settings["bunch"]["design"]["alpha_y"],
        alpha_z=settings["bunch"]["design"]["alpha_z"],
        beta_x=settings["bunch"]["design"]["beta_x"],
        beta_y=settings["bunch"]["design"]["beta_y"],
        beta_z=settings["bunch"]["design"]["beta_z"],
        eps_x=settings["bunch"]["design"]["eps_x"],
        eps_y=settings["bunch"]["design"]["eps_y"],
        eps_z=settings["bunch"]["design"]["eps_z"],
        mass=bunch.mass(),
        kin_energy=bunch.getSyncParticle().kinEnergy()
    )
    
# Center the bunch.
bunch = pyorbit_sim.bunch_utils.set_centroid(
    bunch, 
    centroid=settings["bunch"]["centroid"],
    verbose=True,
)

# Generate RMS equivalent bunch.
if settings["bunch"]["rms_equiv"] is not None:
    bunch = pyorbit_sim.bunch_utils.generate_rms_equivalent_bunch(
        dist=settings["bunch"]["rms_equiv"],
        bunch=bunch, 
        verbose=True,
    )
        
# Downsample. (Assume the particles were randomly generated to begin with.)
fraction_keep = settings["bunch"]["downsample"]
if fraction_keep and fraction_keep < 1.0:
    print("(rank {}) Downsampling by factor {}.".format(_mpi_rank, 1.0 / fraction_keep))
    n = int(fraction_keep * bunch.getSize())  # on each processor
    for i in reversed(range(n, bunch.getSize())):
        bunch.deleteParticleFast(i)
    bunch.compress()    
        
# Decorrelate the x-x', y-y', z-z' coordinates. (No MPI.)
if settings["bunch"]["decorrelate_x-y-z"]:
    bunch = pyorbit_sim.bunch_utils.decorrelate_x_y_z(bunch, verbose=True)
    
# Set the macro-particle size.
size_global = bunch.getSizeGlobal()
macro_size = settings["bunch"]["intensity"] / size_global
bunch.macroSize(macro_size)
    
# Print bunch parameters and statistics.
bunch_info = pyorbit_sim.bunch_utils.get_info(bunch, display=True)


# Tracking
# --------------------------------------------------------------------------------------

if not settings["output"]["save"]:
    settings["sim"]["save_input_bunch"] = False
    settings["sim"]["save_output_bunch"] = False
    settings["sim"]["stride"]["write_bunch"] = np.inf
    settings["sim"]["stride"]["plot_bunch"] = np.inf

    
# Add particle ids.
if settings["sim"]["particle_ids"]:
    ParticleIdNumber.addParticleIdNumbers(bunch)
    copyCoordsToInitCoordsAttr(bunch)

    
# Create bunch writer.
writer = pyorbit_sim.linac.BunchWriter(
    folder=man.outdir, 
    prefix=man.prefix, 
    index=0,
)       
    
# Create bunch plotter. This does not currently work with MPI. The plotter is disabled
# if not saving output.
def transform(X):
    X[:, :] = X[:, :] * 1000.0
    X = pyorbit_sim.bunch_utils.norm_xxp_yyp_zzp(X, scale_emittance=True)
    X = pyorbit_sim.bunch_utils.slice_sphere(
        X,
        axis=(0, 1, 2, 3),
        rmin=0.0,
        rmax=1.0,
    )
    return X

plotter = pyorbit_sim.plotting.Plotter(
    transform=transform, 
    folder=man.outdir,
    prefix=man.prefix,
    default_save_kws=None, 
    dims=["x", "xp", "y", "yp", "z", "dE"],
)
plot_kws = dict(
    text=True,
    colorbar=True,
    floor=1.0,
    divide_by_max=True,
    profx=True,
    profy=True,
    bins=75,
    norm="log",
)
plotter.add_function(
    pyorbit_sim.plotting.proj2d, 
    axis=(4, 5),
    limits=[(-5.0, 5.0), (-5.0, 5.0)],
    save_kws=dict(dpi=200), 
    name=None, 
    **plot_kws
)

    
# Create bunch monitor.
monitor = pyorbit_sim.linac.Monitor(
    position_offset=0.0,  # will be set automatically in `pyorbit_sim.linac.track`.
    stride=settings["sim"]["stride"],
    writer=writer,
    plotter=plotter,
    track_rms=True,
    filename=man.get_filename("history.dat"),
    rf_frequency=linac.rf_frequency,
    verbose=True,
)

# Record synchronous particle time of arrival at each accelerating cavity.
if _mpi_rank == 0:
    print("Tracking design bunch...")
design_bunch = lattice.trackDesignBunch(bunch)
if _mpi_rank == 0:
    print("Design bunch tracking complete.")
    
# Check the synchronous particle time. This could be wrong if start != 0 and
# if the bunch was loaded from a file without a PyORBIT header.
pyorbit_sim.linac.check_sync_part_time(
    bunch, 
    lattice, 
    start=settings["sim"]["start"], 
    set_design=settings["sim"]["set_design_sync_time"],
)
    
# Save input bunch.
if settings["sim"]["save_input_bunch"]:
    node_name = settings["sim"]["start"]
    if node_name is None or type(node_name) is not str:
        node_name = "START"
    writer.action(bunch, node_name)
    
    
# Track
params_dict = pyorbit_sim.linac.track(
    bunch, 
    lattice, 
    monitor=monitor, 
    start=settings["sim"]["start"], 
    stop=settings["sim"]["stop"], 
    verbose=True
)
    
    
# # # Save losses vs. position.
# # aprt_nodes_losses = GetLostDistributionArr(aperture_nodes, params_dict["lostbunch"])
# # if _mpi_rank == 0:
# #     print("Total loss = {:.2e}".format(sum([loss for (node, loss) in aprt_nodes_losses])))
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

sys.exit()