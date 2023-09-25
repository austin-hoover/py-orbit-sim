"""SNS linac simulation."""
from __future__ import print_function
import argparse
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

from sns_linac import SNS_LINAC



# Parse command line arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser("sim")

# Input/output paths
parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--xml-file", type=str, default="sns_linac.xml")
parser.add_argument("--rf-file", type=str, default="sns_rf_fields.xml")
    
# Saving
parser.add_argument("--save", type=int, default=1)
parser.add_argument("--save-init-coords-attr", type=int, default=0)
parser.add_argument("--save-input-bunch", type=int, default=1)
parser.add_argument("--save-output-bunch", type=int, default=1)
parser.add_argument("--save-lost-bunch", type=int, default=1)
parser.add_argument("--save-losses", type=int, default=1)
parser.add_argument("--save-ids", type=int, default=1)
parser.add_argument("--verbose", type=int, default=1)

parser.add_argument("--stride-update", type=float, default=0.100)
parser.add_argument("--stride-write", type=float, default=float("inf"))
parser.add_argument("--stride-plot", type=float, default=float("inf"))

# Lattice
parser.add_argument("--aprt-trans", type=int, default=1)
parser.add_argument("--aprt-phase", type=int, default=1)
parser.add_argument("--aprt-phase-min", type=float, default=-90.0)
parser.add_argument("--aprt-phase-max", type=float, default=+90.0)
parser.add_argument("--aprt-energy", type=int, default=0)
parser.add_argument("--aprt-energy-min", type=float, default=-0.100)
parser.add_argument("--aprt-energy-max", type=float, default=+0.100)
parser.add_argument("--linac-tracker", type=int, default=1)
parser.add_argument("--max-drift", type=float, default=0.010)
parser.add_argument("--overlap", type=int, default=0)
parser.add_argument("--overlap-zstep", type=float, default=0.002)
parser.add_argument("--rf-freq", type=float, default=402.5e+06)
parser.add_argument("--rf-model", type=str, default="ttf")
parser.add_argument("--seq-start", type=str, default="MEBT")
parser.add_argument("--seq-stop", type=str, default="DTL6")

# Space charge
parser.add_argument("--sc", type=int, default=1)
parser.add_argument("--sc-gridx", type=int, default=64)
parser.add_argument("--sc-gridy", type=int, default=64)
parser.add_argument("--sc-gridz", type=int, default=64)

# Bunch
parser.add_argument("--bunch", type=str, default=None)
parser.add_argument("--charge", type=float, default=-1.0)  # [elementary charge units]
parser.add_argument("--current", type=float, default=0.038)  # [A]
parser.add_argument("--energy", type=float, default=0.0025)  # [GeV]
parser.add_argument("--mass", type=float, default=0.939294)  # [GeV / c^2]
parser.add_argument("--dist", type=str, default="wb", choices=["kv", "wb", "gs"])
parser.add_argument("--n-parts", type=int, default=None)
parser.add_argument("--decorr", type=int, default=0)
parser.add_argument("--rms-equiv", type=int, default=0)
parser.add_argument("--mean_x", type=float, default=None)
parser.add_argument("--mean_y", type=float, default=None)
parser.add_argument("--mean_z", type=float, default=None)
parser.add_argument("--mean_xp", type=float, default=None)
parser.add_argument("--mean_yp", type=float, default=None)
parser.add_argument("--mean_dE", type=float, default=None)
parser.add_argument("--set-sync", type=int, default=1)

# Tracking
parser.add_argument("--start", type=str, default=None)
parser.add_argument("--start-pos", type=float, default=None)
parser.add_argument("--stop", type=str, default=None)
parser.add_argument("--stop-pos", type=float, default=None)

args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------
file_path = os.path.realpath(__file__)
file_dir = os.path.dirname(file_path)

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

# Create output directory and save script info.
man = ScriptManager(outdir=output_dir, filepath=file_path)
if args.save and _mpi_rank == 0:
    man.make_dirs()
    log = man.get_logger(save=args.save, disp=True)
    for key, val in man.get_info().items():
        log.info("{} {}".format(key, val))
    log.info(args)
    man.save_script_copy()


# Lattice 
# --------------------------------------------------------------------------------------

linac = SNS_LINAC(rf_frequency=args.rf_freq)

lattice = linac.initialize(
    xml_filename=os.path.join(input_dir, args.xml_file),
    sequence_start=args.seq_start,
    sequence_stop=args.seq_stop,
    max_drift_length=args.max_drift,
    verbose=True,
)

rf_gap_models = {
    "ttf": RfGapTTF,
}
linac.set_rf_gap_model(rf_gap_models[args.rf_model])

if args.aprt_trans:
    linac.add_transverse_aperture_nodes(
        scrape_x=0.042,
        scrape_y=0.042,
        verbose=True
    )
    
if args.aprt_phase:
    phase_min = args.aprt_phase_min
    phase_max = args.aprt_phase_max
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
        step=0.050,
        verbose=True,
    )
if args.aprt_energy:
    energy_min = args.aprt_energy_min
    energy_max = args.aprt_energy_max
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
        step=0.1,
        verbose=True,
    )

if args.overlap:
    linac.set_overlapping_rf_and_quad_fields(
        sequences=linac.sequences,
        z_step=args.overlap_zstep,
        fields_filename=os.path.join(input_dir, args.rf_file),
    )
                
linac.add_space_charge_nodes(
    solver="FFT",
    grid_size=(args.sc_gridx, args.sc_gridy, args.sc_gridz),
    path_length_min=args.max_drift,
    verbose=True,
)
for sc_node in linac.sc_nodes:
    sc_node.setCalculationOn(args.sc)
    
linac.set_linac_tracker(args.linac_tracker)

if args.save:
    linac.save_node_positions(man.get_filename("lattice_nodes.txt"))
    linac.save_lattice_structure(man.get_filename("lattice_structure.txt"))

lattice = linac.lattice


# Bunch
# ------------------------------------------------------------------------------

dist_constructors = {
    "kv": KVDist3D, 
    "wb": WaterBagDist3D, 
    "gs": GaussDist3D,
}
dist_constructor = dist_constructors[args.dist]

bunch = Bunch()
bunch.mass(args.mass)
bunch.charge(args.charge)
bunch.getSyncParticle().kinEnergy(args.energy)

if args.bunch is None:
    bunch = pyorbit_sim.bunch_utils.generate_norm_twiss(
        dist_constructor=dist_constructor,
        n=args.n_parts,
        bunch=bunch,
        verbose=args.verbose,
        **config["bunch"]["twiss"]
    )
else:
    bunch = pyorbit_sim.bunch_utils.load(
        filename=args.bunch,
        bunch=bunch,
        verbose=args.verbose,
    )
    
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

if args.rms_equiv:
    bunch = pyorbit_sim.bunch_utils.generate_rms_equivalent_bunch(
        dist=dist,
        bunch=bunch,
        verbose=args.verbose,
    )

if args.n_parts:
    bunch = pyorbit_sim.bunch_utils.downsample(
        bunch,
        n=args.n_parts,
        method="first",
        conserve_intensity=True,
        verbose=args.verbose,
    )

if args.decorr:
    bunch = pyorbit_sim.bunch_utils.decorrelate_x_y_z(bunch, verbose=args.verbose)

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
    stride["write"] = float("inf")
    stride["plot"] = float("inf")
        
    
def transform_normalize(X):
    return pyorbit_sim.cloud.norm_xxp_yyp_zzp(X, scale_emittance=True)

def transform_slice_transverse_ellipsoid(X):
    return pyorbit_sim.cloud.slice_sphere(
        X,
        axis=(0, 1, 2, 3),
        rmin=0.0,
        rmax=1.0,
    )

plotter = pyorbit_sim.plotting.Plotter(
    transform=transform_normalize, 
    outdir=man.outdir,
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
    bins=100,
    norm="log",
)
save_kws=dict(dpi=200)

plotter.add_function(
    pyorbit_sim.plotting.proj2d, 
    transform=None,
    axis=(4, 5),
    limits=[(-5.0, 5.0), (-5.0, 5.0)],
    save_kws=save_kws,
    name="proj2d_zdE", 
    **plot_kws
)
plotter.add_function(
    pyorbit_sim.plotting.proj2d, 
    transform=transform_slice_transverse_ellipsoid,
    axis=(4, 5),
    limits=[(-5.0, 5.0), (-5.0, 5.0)],
    save_kws=save_kws,
    name="proj2d_zdE_slice_xxpyyp", 
    **plot_kws
)


# Create bunch writer.
writer = pyorbit_sim.linac.BunchWriter(outdir=man.outdir, index=0)       

    
# Create bunch monitor.
monitor = pyorbit_sim.linac.Monitor(
    position_offset=0.0,  # will be set automatically in `pyorbit_sim.linac.track`.
    stride=stride,
    writer=writer,
    plotter=plotter,
    track_rms=True,
    filename=(man.get_filename("history.dat") if args.save else None),
    rf_frequency=linac.rf_frequency,
    verbose=True,
)


# Write particle ids in column 7 of each bunch file.
if args.save_ids:
    ParticleIdNumber.addParticleIdNumbers(bunch)
    
# Write initial 6D coordinates to columns 8-13 of each bunch file.
if args.save_init_coords_attr:
    copyCoordsToInitCoordsAttr(bunch)
    
    
# Tracking
# --------------------------------------------------------------------------------------

start = args.start
if args.start_pos is not None:
    start = args.start_pos
stop = args.stop
if args.stop_pos is not None:
    stop = args.stop_pos

if _mpi_rank == 0:
    print("Tracking design bunch...")
design_bunch = lattice.trackDesignBunch(bunch)
if _mpi_rank == 0:
    print("Design bunch tracking complete.")

pyorbit_sim.linac.check_sync_part_time(
    bunch, 
    lattice, 
    start=start,
    set_design=args.set_sync,
    verbose=True
)
    
if args.save and args.save_input_bunch:
    node_name = start
    if node_name is None or type(node_name) is not str:
        node_name = "START"
    writer.action(bunch, node_name)
    
params_dict = pyorbit_sim.linac.track(
    bunch, lattice, monitor=monitor, start=start, stop=stop, verbose=True
)
        
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
    
if args.save and args.save_lost_bunch:
    filename = man.get_filename("lostbunch.dat")
    if _mpi_rank == 0:
        print("Writing lostbunch to file {}".format(filename))
    lostbunch = params_dict["lostbunch"]
    lostbunch.dumpBunch(filename)

if args.save and args.save_output_bunch:
    node_name = stop
    if node_name is None or type(node_name) is not str:
        node_name = "STOP"
    writer.action(bunch, node_name)

if _mpi_rank == 0:
    print("SIMULATION COMPLETE")
    print("outdir = {}".format(man.outdir))
    print("timestamp = {}".format(man.timestamp))