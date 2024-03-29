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

import pyorbit_sim.bunch_utils
import pyorbit_sim.linac
import pyorbit_sim.plotting
from pyorbit_sim.misc import lorentz_factors
from pyorbit_sim.utils import ScriptManager

from sns_linac import SNS_LINAC


# Setup
# --------------------------------------------------------------------------------------

# These are defined here for convenience. We can also define them at the start of each 
# section, along with the settings.
switches = {
    "save": True,
    "lattice": {
        "apertures": {
            "transverse": True,
            "phase": True,
            "energy": False,
        },
        "linac_tracker": True,
        "overlapping_fields": False,
        "space_charge": True,
    },
    "bunch": {
        "rms_equivalent_dist": False,
        "decorrelate_x-y-z": False,
        "set_design_sync_time": True,
    },
    "diagnostics": {
        "save_input_bunch": True,
        "save_output_bunch": True,
        "save_init_coords_attr": False,
        "save_particle_ids": True,
        "save_lostbunch": True,  
        "save_losses": True,
    },
}

# Set input directories.
output_dir = "/home/46h/sim_data/"  # parent directory for output
file_dir = os.path.dirname(os.path.realpath(__file__))  # directory of this file
input_dir = os.path.join(file_dir, "data_input")  # lattice input data

# MPI
_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

# Broadcast timestamp from MPI rank 0.
main_rank = 0
datestamp = time.strftime("%Y-%m-%d")
timestamp = time.strftime("%y%m%d%H%M%S")
datestamp = orbit_mpi.MPI_Bcast(datestamp, orbit_mpi.mpi_datatype.MPI_CHAR, main_rank, _mpi_comm)
timestamp = orbit_mpi.MPI_Bcast(timestamp, orbit_mpi.mpi_datatype.MPI_CHAR, main_rank, _mpi_comm)

# Create output directory and save script info.
man = ScriptManager(
    datadir=output_dir,
    path=pathlib.Path(__file__), 
    timestamp=timestamp,
    datestamp=datestamp,
)
if _mpi_rank == 0 and switches["save"]:
    man.make_outdir()
    man.save_info()
    man.save_script_copy()
    pprint(man.get_info())


# Lattice 
# --------------------------------------------------------------------------------------

# Create the lattice.
linac = SNS_LINAC(rf_frequency=402.5e+06)
lattice = linac.initialize(
    xml_filename=os.path.join(input_dir, "sns_linac.xml")
    sequence_start="MEBT",
    sequence_stop="DTL6",
    max_drift_length=0.010,
    verbose=True,
)

# Set RF gap model.
linac.set_rf_gap_model(RfGapTTF)

# Set overlapping fields model.
if switches["lattice"]["overlapping_fields"]:    
    linac.set_overlapping_rf_and_quad_fields(
        sequences=linac.sequences,
        z_step=0.002,
        xml_filename=os.path.join(input_dir, "sns_rf_fields.xml"),
    )
    
# Space charge
linac.add_space_charge_nodes(
    solver="FFT",
    grid_size=(128, 128, 128),
    path_length_min=0.010,
    verbose=True,
)
for sc_node in linac.sc_nodes:
    sc_node.setCalculationOn(switches["lattice"]["space_charge"])
            
# Apertures
if switches["lattice"]["apertures"]["transverse"]:
    linac.add_transverse_aperture_nodes(
        scrape_x=0.042,
        scrape_y=0.042,
        verbose=True
    )
if switches["lattice"]["apertures"]["phase"]:
    phase_min = -90.0  # [deg]
    phase_max = +90.0  # [deg]
    linac.add_phase_aperture_nodes(
        classes=[
            BaseRF_Gap, 
            AxisFieldRF_Gap, 
            AxisField_and_Quad_RF_Gap,
            Quad, 
            OverlappingQuadsNode,
        ],
        phase_min=phase_min,
        phase_max=phase_max,
        verbose=True,
    )
    linac.add_phase_aperture_nodes_drifts(
        phase_min=phase_min,
        phase_max=phase_max,
        start=0.0,
        stop=4.0,
        step=0.050,  # [m]
        verbose=True,
    )
if switches["lattice"]["apertures"]["energy"]:
    energy_min = -0.100  # [GeV]
    energy_max = +0.100  # [GeV]
    linac.add_energy_aperture_nodes(
        classes=[
            BaseRF_Gap, 
            AxisFieldRF_Gap, 
            AxisField_and_Quad_RF_Gap,
            Quad, 
            OverlappingQuadsNode,
        ],
        energy_min=energy_min,
        energy_max=energy_max,
        verbose=True,
    )
    linac.add_energy_aperture_nodes_drifts(
        energy_min=energy_min,
        energy_max=energy_max,
        step=0.1,  # [m]
        verbose=True,
    )
    
# Set linac tracker.
linac.set_linac_tracker(switches["lattice"]["linac_tracker"])

# Save lattice info.
if switches["save"]:
    linac.save_node_positions(man.get_filename("lattice_nodes.txt"))
    linac.save_lattice_structure(man.get_filename("lattice_structure.txt"))


lattice = linac.lattice


# Bunch
# --------------------------------------------------------------------------------------

# Settings
filename = os.path.join(
    "/home/46h/projects/BTF/sim/SNS_RFQ/parmteq/2021-01-01_benchmark/data/",
    "bunch_RFQ_output_8.56e+06_decorr_x-y-z.dat",
)
mass = 0.939294  # [GeV / c^2]
charge = -1.0  # [elementary charge units]
kin_energy = 0.0025  # [GeV]
current = 0.042  # [A]
n_parts = None  # max number of particles

# Initialize the bunch.
bunch = Bunch()
bunch.mass(mass)
bunch.charge(charge)
bunch.getSyncParticle().kinEnergy(kin_energy)

# Load the bunch coordinates or generate from Twiss parameters.
if filename is not None:
    pyorbit_sim.bunch_utils.load(
        filename=filename,
        bunch=bunch,
        verbose=True,
    )
else:
    if n_parts is None:
        n_parts = int(1e4)
    bunch = pyorbit_sim.bunch_utils.generate_norm_twiss(
        dist=WaterBagDist3D,
        n=n_parts,
        bunch=bunch,
        verbose=True,
        alpha_x=-1.9620,
        alpha_y=1.7681,
        alpha_z=-0.0196,
        beta_x=0.1831,
        beta_y=0.1620,
        beta_z=0.5844,
        eps_x=0.21e-06,
        eps_y=0.21e-06,
        eps_z=0.24153e-06,
    )
    
# Set the bunch centroid.
bunch = pyorbit_sim.bunch_utils.set_centroid(
    bunch, 
    centroid=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    verbose=False,
)

# Generate an rms-equivalent bunch.
if switches["bunch"]["rms_equivalent_dist"]:
    bunch = pyorbit_sim.bunch_utils.generate_rms_equivalent_bunch(
        dist=WaterbagDist3D,
        bunch=bunch, 
        verbose=True,
    )
            
# Downsample. (Assume the particles were randomly generated to begin with.)
if n_parts is not None:
    size_global = bunch.getSizeGlobal()
    if n_parts < size_global:
        bunch = pyorbit_sim.bunch_utils.downsample(
            bunch, 
            n=n_parts,
            method="first", 
            verbose=True,
        )

# Decorrelate the x-x', y-y', z-z' coordinates. (No MPI.)
if switches["bunch"]["decorrelate_x-y-z"]:
    bunch = pyorbit_sim.bunch_utils.decorrelate_x_y_z(bunch, verbose=True)
        
# Set the macro-particle size.
intensity = pyorbit_sim.bunch_utils.get_intensity(current, linac.rf_frequency)
size_global = bunch.getSizeGlobal()
macro_size = intensity / size_global
bunch.macroSize(macro_size)

# Print the bunch parameters and statistics.
info = pyorbit_sim.bunch_utils.get_info(bunch)
if _mpi_rank == 0:
    print("Bunch info:")
    pprint(info)

    
# Diagnostics
# --------------------------------------------------------------------------------------

stride = {
    "update": 0.100,  # [m]
    "write_bunch": 6.0,  # [m]
    "plot_bunch": np.inf,  # [m]
}
if not switches["save"]:
    stride["write_bunch"] = np.inf
    stride["plot_bunch"] = np.inf
        
    
# Create bunch plotter.
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


# Create bunch writer.
writer = pyorbit_sim.linac.BunchWriter(
    folder=man.outdir, 
    prefix=man.prefix, 
    index=0,
)       

    
# Create bunch monitor.
monitor = pyorbit_sim.linac.Monitor(
    position_offset=0.0,  # will be set automatically in `pyorbit_sim.linac.track`.
    stride=stride,
    writer=writer,
    plotter=plotter,
    track_rms=True,
    filename=(man.get_filename("history.dat") if switches["save"] else None),
    rf_frequency=linac.rf_frequency,
    verbose=True,
)


# Write particle ids in column 7 of each bunch file.
if switches["diagnostics"]["save_particle_ids"]:
    ParticleIdNumber.addParticleIdNumbers(bunch)
    
# Write initial 6D coordinates to columns 8-13 of each bunch file.
if switches["diagnostics"]["save_init_coords_attr"]:
    copyCoordsToInitCoordsAttr(bunch)
    
    
# Tracking
# --------------------------------------------------------------------------------------

# Settings
start = None  # (node name/position/None)
stop = None  # (node name/position/None)

# Record synchronous particle time of arrival at each accelerating cavity.
if _mpi_rank == 0:
    print("Tracking design bunch...")
design_bunch = lattice.trackDesignBunch(bunch)
if _mpi_rank == 0:
    print("Design bunch tracking complete.")

# Check the synchronous particle time.
pyorbit_sim.linac.check_sync_part_time(
    bunch, 
    lattice, 
    start=start,
    set_design=switches["bunch"]["set_design_sync_time"],
    verbose=True
)
    
# Save the bunch.
if switches["save"] and switches["save_input_bunch"]:
    node_name = start
    if node_name is None or type(node_name) is not str:
        node_name = "START"
    writer.action(bunch, node_name)
    
# Track the bunch.
params_dict = pyorbit_sim.linac.track(
    bunch, 
    lattice, 
    monitor=monitor, 
    start=start,
    stop=stop,
    verbose=True
)
        
# Save loss statistics.
if len(linac.aperture_nodes) > 0:
    aprt_nodes_losses = GetLostDistributionArr(linac.aperture_nodes, params_dict["lostbunch"])
    total_loss = sum([loss for (node, loss) in aprt_nodes_losses])
    if _mpi_rank == 0:
        print("Total loss = {:.2e}".format(total_loss))
    if switches["save"] and switches["save_losses"]:
        filename = man.get_filename("losses.txt")
        if _mpi_rank == 0:
            print("Saving loss vs. node array to {}".format(filename))
        file = open(filename, "w")
        file.write("node position loss\n")
        for (node, loss) in aprt_nodes_losses:
            file.write("{} {} {}\n".format(node.getName(), node.getPosition(), loss))
        file.close()
    
# Save lost bunch.
if switches["save"] and switches["save_lostbunch"]:
    filename = man.get_filename("lostbunch.dat")
    if _mpi_rank == 0:
        print("Writing lostbunch to file {}".format(filename))
    lostbunch = params_dict["lostbunch"]
    lostbunch.dumpBunch(filename)

# Save the bunch.
if switches["save"] and switches["save_output_bunch"]:
    node_name = stop 
    if node_name is None or type(node_name) is not str:
        node_name = "STOP"
    writer.action(bunch, node_name)
    
if _mpi_rank == 0:
    print("timestamp = {}".format(man.timestamp))
