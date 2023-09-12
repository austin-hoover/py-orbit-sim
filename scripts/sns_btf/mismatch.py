"""Print the beam size at the FODO quads.

This script is used to quickly compute mismatched optics.
"""
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
from orbit.bunch_generators import GaussDist3D
from orbit.bunch_generators import KVDist3D
from orbit.bunch_generators import WaterBagDist3D
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.utils import consts
import orbit_mpi

from sns_btf import SNS_BTF

import pyorbit_sim


# Parse arguments
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--save", type=int, default=0)
parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--xml", type=str, default="xml/btf_lattice_straight.xml")
parser.add_argument("--coef", type=str, default="magnets/default_i2gl_coeff_straight.csv")
parser.add_argument("--mstate", type=str, default=None)
parser.add_argument("--quads", type=str, default="magnets/230905100947-quad_strengths_matched.dat")

parser.add_argument("--apertures", type=int, default=1)
parser.add_argument("--linac_tracker", type=int, default=1)
parser.add_argument("--max_drift", type=float, default=0.010)
parser.add_argument("--overlap", type=int, default=1)
parser.add_argument("--rf_freq", type=float, default=402.5e+06)

parser.add_argument("--bunch", type=str, default=None)
parser.add_argument("--charge", type=float, default=-1.0)  # [elementary charge units]
parser.add_argument("--current", type=float, default=0.042)  # [A]
parser.add_argument("--energy", type=float, default=0.0025)  # [GeV]
parser.add_argument("--mass", type=float, default=0.939294)  # [GeV / c^2]
parser.add_argument("--dist", type=str, default="wb", choices=["kv", "wb", "gs"])
parser.add_argument("--nparts", type=int, default=10000)
parser.add_argument("--decorr", type=int, default=0)
parser.add_argument("--rms_equiv", type=int, default=0)
parser.add_argument("--mean_x", type=float, default=None)
parser.add_argument("--mean_y", type=float, default=None)
parser.add_argument("--mean_z", type=float, default=None)
parser.add_argument("--mean_xp", type=float, default=None)
parser.add_argument("--mean_yp", type=float, default=None)
parser.add_argument("--mean_dE", type=float, default=None)

parser.add_argument("--amp", type=float, default=0.0)

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
man = pyorbit_sim.utils.ScriptManager(outdir=output_dir, filepath=file_path)
if args.save and _mpi_rank == 0:
    man.make_dirs()
    logger = man.get_logger(save=args.save, disp=True)
    for key, val in man.get_info().items():
        logger.info("{} {}".format(key, val))
    logger.info(args)
    man.save_script_copy()
    
    
# Lattice
# ------------------------------------------------------------------------------
linac = SNS_BTF(
    coef_filename=os.path.join(input_dir, args.coef), 
    rf_frequency=args.rf_freq
)
linac.init_lattice(
    xml_filename=os.path.join(input_dir, args.xml),
    sequences=[
        "MEBT1",
        "MEBT2",
    ],
    max_drift_length=args.max_drift,
)

if args.mstate:
    filename = os.path.join(input_dir, args.mstate)
    linac.update_quads_from_mstate(filename, value_type="current")
    
if args.quads:
    filename = os.path.join(input_dir, args.quads)
    linac.set_quads_from_file(filename, comment="#")

if args.overlap:
    linac.set_overlapping_pmq_fields(z_step=args.max_drift)
    
linac.add_uniform_ellipsoid_space_charge_nodes(
    n_ellipsoids=5,
    path_length_min=args.max_drift,
)

if args.apertures:
    linac.add_aperture_nodes(drift_step=0.1)

if args.linac_tracker:
    linac.set_linac_tracker(args.linac_tracker)

lattice = linac.lattice
    
    
# Bunch
# ------------------------------------------------------------------------------

def_bunch_filename = "/home/46h/projects/btf/sim/sns_rfq/parmteq/2021-01-01_benchmark/data/bunch_RFQ_output_1.00e+05.dat"
dists = {
    "kv": KVDist3D, 
    "wb": WaterBagDist3D, 
    "gs": GaussDist3D,
}
dist = dists[args.dist]

bunch = Bunch()
bunch.mass(args.mass)
bunch.charge(args.charge)
bunch.getSyncParticle().kinEnergy(args.energy)

if args.bunch == "design":
    bunch = pyorbit_sim.bunch_utils.generate_norm_twiss(
        dist=dist,
        n=args.nparts,
        bunch=bunch,
        **config["bunch"]["twiss"]
    )
else:
    filename = args.bunch
    if filename is None:
        filename = def_bunch_filename
    bunch = pyorbit_sim.bunch_utils.load(
        filename=filename,
        bunch=bunch,
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
)

if args.rms_equiv:
    bunch = pyorbit_sim.bunch_utils.generate_rms_equivalent_bunch(
        dist=dist,
        bunch=bunch,
    )

if args.nparts:
    bunch = pyorbit_sim.bunch_utils.downsample(
        bunch,
        n=args.nparts,
        method="first",
        conserve_intensity=True,
    )

if args.decorr:
    bunch = pyorbit_sim.bunch_utils.decorrelate_x_y_z(bunch)

intensity = pyorbit_sim.bunch_utils.get_intensity(args.current, linac.rf_frequency)
bunch_size_global = bunch.getSizeGlobal()
macro_size = intensity / bunch_size_global
bunch.macroSize(macro_size)


# Tracking
# --------------------------------------------------------------------------------------

# Perturb the lattice.
np.random.seed(0)
node_names = ["MEBT:QV09", "MEBT:QH10"]
amp = args.amp
for name in node_names:
    node = lattice.getNodeForName(name)
    kappa = node.getParam("dB/dr")
    delta = amp * kappa
    node.setParam("dB/dr", kappa + delta)


class Monitor:
    def __init__(self, nodes):
        self.nodes = nodes
    
    def action(self, params_dict):
        node = params_dict["node"]
        if node not in self.nodes:
            return
        bunch = params_dict["bunch"]
        twiss_analysis = BunchTwissAnalysis()
        twiss_analysis.computeBunchMoments(bunch, 2, 0, 0)
        x_rms = 1000.0 * np.sqrt(twiss_analysis.getCorrelation(0, 0))
        y_rms = 1000.0 * np.sqrt(twiss_analysis.getCorrelation(2, 2))
        logger.info("xrms={:0.3f} yrms={:0.3f} node={}".format(x_rms, y_rms, node.getName()))
    
        
monitor = Monitor([lattice.getNodeForName(name) for name in linac.quad_names_fodo])
action_container = AccActionsContainer()
action_container.addAction(monitor.action, AccActionsContainer.ENTRANCE)

lattice.trackBunch(
    bunch,
    index_start=0,
    index_stop=-1,
    actionContainer=action_container,
)


if args.save:
    filename = man.get_filename("quad_strengths.dat")
    print("Saving file {}".format(filename))
    linac.save_quad_strengths(filename)