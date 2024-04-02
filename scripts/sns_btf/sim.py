"""SNS BTF simulation."""
from __future__ import print_function
import argparse
import math
import os
import pathlib
import pickle
from pprint import pprint
import shutil
import sys
import time

import numpy as np
import yaml

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

from sns_btf import SNS_BTF


# Parse command line arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser("sim")

# Input/output paths
parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--xml", type=str, default="xml/btf_lattice_straight.xml")
parser.add_argument("--coef", type=str, default="magnets/default_i2gl_coeff_straight.csv")
parser.add_argument("--mstate", type=str, default="mstate/Snapshot_20240111_173323.mstate")
parser.add_argument("--quads", type=str, default=None)
    
# Saving
parser.add_argument("--save", type=int, default=1)
parser.add_argument("--save_init_coords_attr", type=int, default=0)
parser.add_argument("--save_input_bunch", type=int, default=1)
parser.add_argument("--save_output_bunch", type=int, default=1)
parser.add_argument("--save_lost_bunch", type=int, default=1)
parser.add_argument("--save_losses", type=int, default=1)
parser.add_argument("--save_ids", type=int, default=1)
parser.add_argument("--verbose", type=int, default=1)

parser.add_argument("--stride_update", type=float, default=0.010)  # [m]
parser.add_argument("--stride_write", type=float, default=float("inf"))  # [m]
parser.add_argument("--stride_plot", type=float, default=float("inf"))  # [m]

# Lattice
parser.add_argument("--apertures", type=int, default=1)
parser.add_argument("--linac_tracker", type=int, default=1)
parser.add_argument("--max_drift", type=float, default=0.010)
parser.add_argument("--overlap", type=int, default=0)
parser.add_argument("--set_sync_time", type=int, default=1)
parser.add_argument("--rf", type=float, default=402.5e+06)

# Space charge
parser.add_argument("--spacecharge", type=int, default=1)
parser.add_argument("--gridx", type=int, default=64)
parser.add_argument("--gridy", type=int, default=64)
parser.add_argument("--gridz", type=int, default=64)
parser.add_argument("--bunches", type=int, default=3)

# Bunch

# If None, use default bunch filename defined below.
# If "design", use the design Twiss parameters defined below with the 
# distribution specified by args.dist. 
# Otherwise, specifies the bunch filename.
parser.add_argument("--bunch", type=str, default=None)

parser.add_argument("--charge", type=float, default=-1.0)  # [elementary charge units]
parser.add_argument("--current", type=float, default=0.042)  # [A]
parser.add_argument("--energy", type=float, default=0.0025)  # [GeV]
parser.add_argument("--mass", type=float, default=0.939294)  # [GeV / c^2]

parser.add_argument("--dist", type=str, default="wb", choices=["kv", "wb", "gs"])
parser.add_argument("--n", type=int, default=None, help="number of particles")
parser.add_argument("--decorr", type=int, default=0)
parser.add_argument("--rms_equiv", type=int, default=0)
parser.add_argument("--mean_x", type=float, default=0.0)
parser.add_argument("--mean_y", type=float, default=0.0)
parser.add_argument("--mean_z", type=float, default=0.0)
parser.add_argument("--mean_xp", type=float, default=0.0)
parser.add_argument("--mean_yp", type=float, default=0.0)
parser.add_argument("--mean_dE", type=float, default=0.0)


# Tracking
parser.add_argument("--start", type=str, default=None)
parser.add_argument("--start_pos", type=float, default=None)
parser.add_argument("--stop", type=str, default=None)
parser.add_argument("--stop_pos", type=float, default=None)

args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------
file_dir = os.path.dirname(os.path.realpath(__file__))

# Load config file.
file = open(os.path.join(file_dir, "config.yaml"), "r")
config = yaml.safe_load(file)
file.close()

# Set input/output directories.
input_dir = os.path.join(file_dir, "data_input")
output_dir = os.path.join(file_dir, config["output_dir"])
if args.outdir is not None:
    output_dir = os.path.join(file_dir, args.outdir)

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
if args.save and _mpi_rank == 0:
    man.make_outdir()
    log = man.get_logger(save=args.save, disp=True)
    for key, val in man.get_info().items():
        log.info("{} {}".format(key, val))
    log.info(args)
    man.save_script_copy()


# Lattice
# ------------------------------------------------------------------------------
sequences = [
    "MEBT1",
    "MEBT2",
]

# Initialize the lattice.
linac = SNS_BTF(coef_filename=os.path.join(input_dir, args.coef), rf_frequency=args.rf)
linac.init_lattice(
    xml_filename=os.path.join(input_dir, args.xml),
    sequences=sequences,
    max_drift_length=args.max_drift,
)

# Set optics from file.
if args.mstate:
    filename = os.path.join(input_dir, args.mstate)
    linac.set_quads_from_mstate(filename, parameter="current")
if args.quads:
    filename = os.path.join(input_dir, args.quads)
    linac.set_quads_from_file(filename, comment="#", verbose=args.verbose)

# Overlapping PMQ fields.
if args.overlap:
    linac.set_overlapping_pmq_fields(z_step=args.max_drift, verbose=args.verbose)

# Space charge
if args.spacecharge:
    linac.add_space_charge_nodes(
        grid_size_x=args.gridx,
        grid_size_y=args.gridy,
        grid_size_z=args.gridz,
        path_length_min=args.max_drift,
        n_bunches=args.bunches,
    )

# Apertures
if args.apertures:
    linac.add_aperture_nodes(drift_step=0.1, verbose=args.verbose)

# Tracking module
linac.set_linac_tracker(args.linac_tracker)

# Save lattice info.
if args.save:
    linac.save_node_positions(man.get_filename("lattice_nodes.txt"))
    linac.save_lattice_structure(man.get_filename("lattice_structure.txt"))

lattice = linac.lattice

                    
# Bunch
# ------------------------------------------------------------------------------

# Define default bunch filename.
def_bunch_filename = os.path.join(
    "/home/46h/projects/btf/sim/sns_rfq/parmteq/2021-01-01_benchmark/data/",
    "bunch_RFQ_output_8.56e+06.dat"
)

# Parse arguments.
dists = {
    "kv": KVDist3D, 
    "wb": WaterBagDist3D, 
    "gs": GaussDist3D,
}
dist = dists[args.dist]

# Initialize the bunch.
bunch = Bunch()
bunch.mass(args.mass)
bunch.charge(args.charge)
bunch.getSyncParticle().kinEnergy(args.energy)

# Load the bunch coordinates
if args.bunch == "design":
    bunch = pyorbit_sim.bunch_utils.generate_norm_twiss(
        dist=dist,
        n=args.n,
        bunch=bunch,
        verbose=args.verbose,
        **config["bunch"]["twiss"]
    )
else:
    filename = args.bunch
    if filename is None:
        filename = def_bunch_filename
    bunch = pyorbit_sim.bunch_utils.load(
        filename=filename,
        bunch=bunch,
        verbose=args.verbose,
    )

# Set the bunch centroid.
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
    verbose=args.verbose,
)

# Generate an rms-equivalent bunch.
if args.rms_equiv:
    bunch = pyorbit_sim.bunch_utils.generate_rms_equivalent_bunch(
        dist=dist,
        bunch=bunch,
        verbose=args.verbose,
    )

# Downsample the bunch.
bunch = pyorbit_sim.bunch_utils.downsample(
    bunch,
    n=args.n,
    method="first",
    conserve_intensity=True,
    verbose=args.verbose,
)

# Decorrelate the x-x', y-y', z-z' coordinates. (No MPI.)
if args.decorr:
    bunch = pyorbit_sim.bunch_utils.decorrelate_x_y_z(bunch, verbose=args.verbose)

# Set the macro-particle size.
intensity = pyorbit_sim.bunch_utils.get_intensity(args.current, linac.rf_frequency)
bunch_size_global = bunch.getSizeGlobal()
macro_size = intensity / bunch_size_global
bunch.macroSize(macro_size)


# Diagnostics
# --------------------------------------------------------------------------------------
stride = {
    "update": args.stride_update,
    "write": args.stride_write,
    "plot": args.stride_plot,
}
if not args.save:
    stride["write_bunch"] = np.inf
    stride["plot_bunch"] = np.inf

    
# Bunch plotter
# ------------------------------

# Define transformation to be applied before plotting.
def transform(X):
    X[:, :] *= 1000.0
    X = pyorbit_sim.cloud.norm_xxp_yyp_zzp(X, scale_emittance=True)
    return X

# Create plotter.
plotter = pyorbit_sim.plotting.Plotter(
    transform=transform,
    outdir=man.outdir,
    default_save_kws=None,
    dims=["x", "xp", "y", "yp", "z", "dE"],
)

# Add functions to plotter.
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
    axis=(0, 1),
    limits=[(-5.0, 5.0), (-5.0, 5.0)],
    save_kws=dict(dpi=200),
    name=None,
    **plot_kws
)

## Add plotter nodes. If these are not added, the plotter will be called
## by `monitor` wtih frequency given by `stride`.
# [...]



# Bunch writer
# ------------------------------
writer = pyorbit_sim.linac.BunchWriter(outdir=man.outdir, index=0)

## Add bunch writer nodes. If these are not added, the writer will be called
## by `monitor` wtih frequency given by `stride`.
# [...]

# Write particle ids in column 7 of each bunch file.
if args.save_ids:
    ParticleIdNumber.addParticleIdNumbers(bunch)

# Write initial 6D coordinates to columns 8-13 of each bunch file.
if args.save_init_coords_attr:
    copyCoordsToInitCoordsAttr(bunch)

    
    
# Monitor
# ------------------------------
monitor = pyorbit_sim.linac.Monitor(
    stride=stride,
    writer=writer,
    plotter=plotter,
    track_rms=True,
    filename=(man.get_filename("history.dat") if args.save else None),
    rf_frequency=linac.rf_frequency,
    verbose=args.verbose,
)



# Tracking
# --------------------------------------------------------------------------------------

# Parse arguments.
start = args.start
if args.start_pos is not None:
    start = args.start_pos
stop = args.stop
if args.stop_pos is not None:
    stop = args.stop_pos

# Save the input bunch.
if args.save and args.save_input_bunch:
    node_name = start
    if node_name is None or type(node_name) is not str:
        node_name = "START"
    writer.action(bunch, node_name)
    
# Track the bunch.
params_dict = pyorbit_sim.linac.track(
    bunch, lattice, monitor=monitor, start=start, stop=stop, verbose=args.verbose
)

# Save the loss statistics. (Note: this will save the loss histogram on
# one MPI processor!)
if linac.aperture_nodes:
    aprt_nodes_losses = GetLostDistributionArr(linac.aperture_nodes, params_dict["lostbunch"])
    total_loss = sum([loss for (node, loss) in aprt_nodes_losses])
    if _mpi_rank == 0:
        print("Total loss = {:.2e}".format(total_loss))
    if args.save and args.save_losses:
        filename = man.get_filename("losses.txt")
        if _mpi_rank == 0:
            print("Saving loss vs. node array to {}".format(filename))
        file = open(filename, "w")
        file.write("node position loss\n")
        for node, loss in aprt_nodes_losses:
            file.write("{} {} {}\n".format(node.getName(), node.getPosition(), loss))
        file.close()

# Save the lost bunch.
if args.save and args.save_lost_bunch:
    filename = man.get_filename("lostbunch.dat")
    if _mpi_rank == 0:
        print("Writing lostbunch to file {}".format(filename))
    lostbunch = params_dict["lostbunch"]
    lostbunch.dumpBunch(filename)

# Save the output bunch.
if args.save and args.save_output_bunch:
    node_name = stop
    if node_name is None or type(node_name) is not str:
        node_name = "STOP"
    writer.action(bunch, node_name)

if _mpi_rank == 0:
    print("SIMULATION COMPLETE")
    print("outdir = {}".format(man.outdir))
    print("timestamp = {}".format(man.timestamp))