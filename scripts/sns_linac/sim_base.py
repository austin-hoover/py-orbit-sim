"""SNS linac simulation (base).

This script does not depend on the pyorbit_sim repo; it can be copied and
run elsewhere.

The basic elements of the linac simulation are included, following https://github.com/PyORBIT-Collaboration/examples/blob/master/SNS_Linac/pyorbit_linac_model/pyorbit_sns_linac_mebt_hebt2.py

This script loads a bunch from a file and tracks it through the lattice. It saves:
    * the input bunch
    * the output bunch
    * various scalars vs. position
"""
from __future__ import print_function
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

from bunch import Bunch
from bunch import BunchTwissAnalysis
from linac import BaseRfGap
from linac import MatrixRfGap
from linac import RfGapTTF
from orbit.bunch_generators import GaussDist3D
from orbit.bunch_generators import KVDist3D
from orbit.bunch_generators import TwissContainer
from orbit.bunch_generators import WaterBagDist3D
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.py_linac.lattice import LinacApertureNode
from orbit.py_linac.lattice import LinacEnergyApertureNode
from orbit.py_linac.lattice import LinacPhaseApertureNode
from orbit.py_linac.lattice import AxisFieldRF_Gap
from orbit.py_linac.lattice import AxisField_and_Quad_RF_Gap
from orbit.py_linac.lattice import BaseRF_Gap
from orbit.py_linac.lattice_modifications import Add_quad_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_rfgap_apertures_to_lattice
from orbit.py_linac.lattice_modifications import AddMEBTChopperPlatesAperturesToSNS_Lattice
from orbit.py_linac.lattice_modifications import AddScrapersAperturesToLattice
from orbit.py_linac.lattice_modifications import GetLostDistributionArr
from orbit.py_linac.linac_parsers import SNS_LinacLatticeFactory
from orbit.space_charge.sc3d import setSC3DAccNodes
from orbit.utils import consts
import orbit_mpi
from spacecharge import SpaceChargeCalc3D


# Setup
# --------------------------------------------------------------------------------------

# MPI
_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

# Create output directory.
outdir = "./data_output/"
if not os.path.isdir(outdir):
    os.makedirs(outdir)
    
# Get timestamp for saved files.
timestamp = time.strftime("%y%m%d%H%M%S")
if _mpi_rank == 0:
    print("timestamp = {}".format(timestamp))
    
    
def get_filename(filename):
    """Add output directory path and timestamp to filename."""
    return os.path.join(outdir, "{}_{}".format(timestamp, filename))


# Save copy of script.
shutil.copy(__file__, get_filename(os.path.basename(__file__)))


# Lattice
# --------------------------------------------------------------------------------------

rf_frequency = 402.5e6  # [1/s]
rf_wavelength = consts.speed_of_light / rf_frequency


# Generate SNS linac lattice from XML file.
xml_file_name = "./data_input/sns_linac.xml"
max_drift_length = 0.010  # [m]
sequences = [
    "MEBT",
    "DTL1",
    "DTL2",
    "DTL3",
    # "DTL4",
    # "DTL5",
    # "DTL6",
    # "CCL1",
    # "CCL2",
    # "CCL3",
    # "CCL4",
    # "SCLMed",
    # "SCLHigh",
    # "HEBT1",
    # "HEBT2",
]
sns_linac_factory = SNS_LinacLatticeFactory()
sns_linac_factory.setMaxDriftLength(max_drift_length)
lattice = sns_linac_factory.getLinacAccLattice(sequences, xml_file_name)
if _mpi_rank == 0:
    print("Initialized lattice.")
    print("XML filename = {}".format(xml_file_name))
    print("lattice length = {:.3f} [m])".format(lattice.getLength()))

    
# Save lattice structure to file.
if _mpi_rank == 0:
    # Node start/stop positions.
    file = open(get_filename("lattice_nodes.txt"), "w")
    file.write("node position length\n")
    for node in lattice.getNodes():
        file.write(
            "{} {} {}\n".format(node.getName(), node.getPosition(), node.getLength())
        )
    file.close()
    # Full lattice structure.
    file = open(get_filename("lattice_structure.txt"), "w")
    file.write(lattice.structureToText())
    file.close()

    
# Set the RF gap model.
for rf_gap in lattice.getRF_Gaps():
    rf_gap.setCppGapModel(RfGapTTF())
    
    
# Ignore overlapping quad/RF fields.


# Ignore linac-style quads (use TEAPOT).


# Add space charge nodes.
sc_grid_size_x = 64
sc_grid_size_y = 64
sc_grid_size_z = 64
sc_path_length_min = 0.010  # [m]
sc_calc = SpaceChargeCalc3D(sc_grid_size_x, sc_grid_size_y, sc_grid_size_z)
sc_nodes = setSC3DAccNodes(lattice, sc_path_length_min, sc_calc)
sc_lengths = [sc_node.getLengthOfSC() for sc_node in sc_nodes]
if _mpi_rank == 0:
    print("Added {} space charge nodes".format(len(sc_nodes)))
    print("  min sc_length = {}".format(min(sc_lengths)))
    print("  max sc_length = {}".format(max(sc_lengths)))

    
# Add transverse aperture nodes.
x_size = 0.042  # [m]
y_size = 0.042  # [m]
aperture_nodes = Add_quad_apertures_to_lattice(lattice)
aperture_nodes = Add_rfgap_apertures_to_lattice(lattice, aperture_nodes)
aperture_nodes = AddMEBTChopperPlatesAperturesToSNS_Lattice(lattice, aperture_nodes)
aperture_nodes = AddScrapersAperturesToLattice(lattice, "MEBT_Diag:H_SCRP", x_size, y_size, aperture_nodes)
aperture_nodes = AddScrapersAperturesToLattice(lattice, "MEBT_Diag:V_SCRP", x_size, y_size, aperture_nodes)
n_transverse_aperture_nodes = len(aperture_nodes)
if _mpi_rank == 0:
    print("Added {} transverse aperture nodes.".format(len(aperture_nodes)))


# Add longitudinal (phase, energy) aperture nodes at each RF gap node.
if False:
    node_pos_dict = lattice.getNodePositionsDict()
    for node in lattice.getNodesOfClasses([BaseRF_Gap, AxisFieldRF_Gap, AxisField_and_Quad_RF_Gap]):
        if node.hasParam("aperture") and node.hasParam("aprt_type"):
            position_start, position_stop = node_pos_dict[node]

            aperture_node = LinacPhaseApertureNode(frequency=rf_frequency, name="{}_phase_aprt_out".format(node.getName()))
            aperture_node.setMinMaxPhase(-180.0, 180.0)  # [deg]
            aperture_node.setPosition(position_stop)
            aperture_node.setSequence(node.getSequence())
            node.addChildNode(aperture_node, node.EXIT)
            aperture_nodes.append(aperture_node)

            aperture_node = LinacEnergyApertureNode(name="{}_energy_aprt_out".format(node.getName()))
            aperture_node.setMinMaxEnergy(-0.100, +0.100)  # [GeV]
            aperture_node.setPosition(position_stop)
            aperture_node.setSequence(node.getSequence())
            node.addChildNode(aperture_node, node.EXIT)
            aperture_nodes.append(aperture_node)            
    if _mpi_rank == 0:
        print("Added {} longitudinal aperture nodes.".format(len(aperture_nodes) - n_transverse_aperture_nodes))


# Bunch
# --------------------------------------------------------------------------------------

## Load bunch from file. The mass, energy, time, etc. will be set from the file header.
# bunch_filename = "./data_input/bunch_RFQ_output_8.56e+06.dat"
bunch_filename = "./data_input/bunch_RFQ_output_8.56e+06_decorr_x-y-z.dat"

if _mpi_rank == 0:
    print("Generating bunch from file '{}'.".format(bunch_filename))
bunch = Bunch()
bunch.readBunch(bunch_filename)

bunch_size_global = bunch.getSizeGlobal()
bunch.getSyncParticle().kinEnergy(0.0025)  # [GeV]
bunch.mass(0.939294)  # [GeV / c^2]
bunch.charge(-1)

current = 0.041  # [A]
charge = (current / rf_frequency)
intensity = (current / rf_frequency) / abs(float(bunch.charge()) * consts.charge_electron)
bunch.macroSize(intensity / bunch_size_global)
    
# Center the bunch at the origin in phase space.
# [...]
        
    
# Downsample. Here we assume the particles were randomly generated to begin with, 
# so we just use the first k indices. Note that random selection is not guaranteed
# to preserve the underlying 6D phase space distribution.
fraction_keep = 0.1
if fraction_keep < 1.0:
    size_global_old = bunch.getSizeGlobal()
    macro_size_old = bunch.macroSize()
    n = int(fraction_keep * bunch.getSize())  # on each processor
    print("(rank {}) Downsampling by factor {}.".format(_mpi_rank, 1.0 / fraction_keep))
    for i in reversed(range(n, bunch.getSize())):
        bunch.deleteParticleFast(i)
    bunch.compress()    
    size_global = bunch.getSizeGlobal()
    bunch.macroSize(intensity / size_global_old)
    
    
# Print bunch parameters.
bunch_size_global = bunch.getSizeGlobal()
macro_size = bunch.macroSize()
intensity = bunch_size_global * macro_size
charge = intensity * abs(float(bunch.charge()) * consts.charge_electron)  # [C]
current = charge * rf_frequency  # [C / s]
if _mpi_rank == 0:
    print("Bunch parameters:")
    print("  charge = {}".format(bunch.charge()))
    print("  mass = {} [GeV / c^2]".format(bunch.mass()))
    print("  kinetic energy = {} [GeV]".format(bunch.getSyncParticle().kinEnergy()))
    print("  macrosize = {}".format(bunch.macroSize()))
    print("  size (local) = {:.2e}".format(bunch.getSize()))
    print("  size (global) = {:.2e}".format(bunch_size_global)) 
    print("  current = {:.5f} [mA]".format(1000.0 * current))

# Print bunch Twiss parameters.
twiss_analysis = BunchTwissAnalysis()
twiss_analysis.analyzeBunch(bunch)
(alpha_x, beta_x, _, eps_x) = twiss_analysis.getTwiss(0)
(alpha_y, beta_y, _, eps_y) = twiss_analysis.getTwiss(1)
(alpha_z, beta_z, _, eps_z) = twiss_analysis.getTwiss(2)
if _mpi_rank == 0:
    print("Twiss parameters:")
    print("  alpha_x = {}".format(alpha_x))
    print("  alpha_y = {}".format(alpha_y))
    print("  alpha_z = {}".format(alpha_z))
    print("  beta_x = {}".format(beta_x))
    print("  beta_y = {}".format(beta_y))
    print("  beta_z = {}".format(beta_z))
    print("  eps_x = {} [mm * mrad]".format(1.0e6 * eps_x))
    print("  eps_y = {} [mm * mrad]".format(1.0e6 * eps_y))
    print("  eps_z = {} [mm * MeV]".format(1.0e6 * eps_z))
    
# Print bunch centroid.
if _mpi_rank == 0:
    print("Centroid coordinates:")
dims = ["x", "xp", "y", "yp", "z", "dE"]
units = ["m", "rad", "m", "rad", "m", "GeV"]
for i, (dim, unit) in enumerate(zip(dims, units)):
    mean = twiss_analysis.getAverage(i)
    if _mpi_rank == 0:
        print("  <{}> = {:.3e} [{}]".format(dim, mean, unit))
    

# Tracking
# --------------------------------------------------------------------------------------

# Set start node.
start_node = lattice.getNodes()[0]
start_node_name = start_node.getName()
start_node_index = lattice.getNodeIndex(start_node)

# Set stop node.
stop_node = lattice.getNodes()[-1]
stop_node_name = stop_node.getName()
stop_node_index = lattice.getNodeIndex(stop_node)


# Record synchronous particle time of arrival at each accelerating cavity.
if _mpi_rank == 0:
    print("Tracking design bunch...")
design_bunch = lattice.trackDesignBunch(bunch)
    
    
# Save input bunch.
filename = get_filename("bunch_0_{}.dat".format(start_node_name))
if _mpi_rank == 0:
    print("Saving bunch to file {}".format(filename))
bunch.dumpBunch(filename)
    
    
# Add actions to action container. 


class Monitor:    
    def __init__(self, start_position=0.0, stride=0.100):
        self.index = 0
        self.stride = stride
        self.position = self.start_position = start_position
        self.start_time = None
        self.history = dict()
        keys = [
            "position",
            "node",
            "n_parts",
            "gamma",
            "beta",
            "energy",
            "x_rms",
            "y_rms",
            "z_rms",
            "z_rms_deg",
            "z_to_phase_coeff",
        ]
        for i in range(6):
            keys.append("mean_{}".format(i))
        for i in range(6):
            for j in range(i + 1):
                keys.append("cov_{}-{}".format(j, i))
        for key in keys:
            self.history[key] = []
        
    def action(self, params_dict):
        # Check the position and decide if we should proceed.
        position = params_dict["path_length"] + self.start_position
        if (position - self.position) < self.stride:
            return
        self.position = position
        
        # Check the time.
        if self.start_time is None:
            self.start_time = time.clock()
        time_ellapsed = time.clock() - self.start_time

        # Get the bunch from the parameter dictionary.
        bunch = params_dict["bunch"]
        node = params_dict["node"]
        n_parts = bunch.getSizeGlobal()
        
        # Record position, energy, etc.
        if _mpi_rank == 0:
            self.history["position"].append(position)
            self.history["node"].append(node.getName())
            self.history["n_parts"].append(n_parts)
            self.history["gamma"].append(bunch.getSyncParticle().gamma())
            self.history["beta"].append(bunch.getSyncParticle().beta())
            self.history["energy"].append(bunch.getSyncParticle().kinEnergy())
            
        # Record first- and second-order moments.
        bunch_twiss_analysis = BunchTwissAnalysis()
        order = 2
        dispersion_flag = False
        emit_norm_flag = False
        bunch_twiss_analysis.computeBunchMoments(bunch, order, dispersion_flag, emit_norm_flag)
        for i in range(6):
            key = "mean_{}".format(i)
            value = bunch_twiss_analysis.getAverage(i)
            if _mpi_rank == 0:
                self.history[key].append(value)
        for i in range(6):
            for j in range(i + 1):
                key = "cov_{}-{}".format(j, i)
                value = bunch_twiss_analysis.getCorrelation(j, i)
                if _mpi_rank == 0:
                    self.history[key].append(value)

        # Record rms sizes.
        if _mpi_rank == 0:
            x_rms = np.sqrt(self.history["cov_0-0"][-1])
            y_rms = np.sqrt(self.history["cov_2-2"][-1])
            z_rms = np.sqrt(self.history["cov_4-4"][-1])
            z_to_phase_coeff = -360.0 / (bunch.getSyncParticle().beta() * rf_wavelength)
            z_rms_deg = -z_to_phase_coeff * z_rms
            self.history["x_rms"].append(x_rms)
            self.history["y_rms"].append(y_rms)
            self.history["z_rms"].append(z_rms)
            self.history["z_rms_deg"].append(z_rms_deg)
            self.history["z_to_phase_coeff"].append(z_to_phase_coeff)

        # Print update statement.
        if _mpi_rank == 0:
            fstr = "{:>5} | {:>10.2f} | {:>7.3f} | {:>8.4f} | {:>9.3f} | {:>9.3f} | {:>10.3f} | {:<9.3e} | {} "
            if self.index == 0:
                print(
                    "{:<5} | {:<10} | {:<7} | {:<8} | {:<5} | {:<9} | {:<10} | {:<9} | {}"
                    .format("index", "time [s]", "s [m]", "T [MeV]", "xrms [mm]", "yrms [mm]", "zrms [deg]", "nparts", "node")
                )
                print(115 * "-")
            print(
                fstr.format(
                    self.index,
                    time_ellapsed,
                    position,
                    1000.0 * bunch.getSyncParticle().kinEnergy(),
                    1000.0 * x_rms,
                    1000.0 * y_rms,
                    z_rms_deg,
                    n_parts,
                    node.getName(),
                )
            )
        self.index += 1
        
    def write_history(self, filename=None, delimiter=","):
        """Write history array to file."""
        keys = list(self.history)
        data = np.array([self.history[key] for key in keys]).T
        df = pd.DataFrame(data=data, columns=keys)
        df.to_csv(filename, sep=delimiter, index=False)
        return df

        
monitor = Monitor(
    start_position=(start_node.getPosition() - 0.5 * start_node.getLength()),
    stride=0.100,
)
action_container = AccActionsContainer()
action_container.addAction(monitor.action, AccActionsContainer.EXIT)


# Track the bunch.
params_dict = dict()
params_dict["bunch"] = bunch
params_dict["lostbunch"] = Bunch()
lattice.trackBunch(
    bunch,
    paramsDict=params_dict,
    actionContainer=action_container,
    index_start=start_node_index,
    index_stop=stop_node_index,
)


# Save scalar history.
monitor.write_history(get_filename("history.dat"))
        
# Save output bunch.
filename = get_filename("bunch_1_{}.dat".format(stop_node_name))
if _mpi_rank == 0:
    print("Saving bunch to file {}".format(filename))
bunch.dumpBunch(filename)

# Save loss array.
aperture_nodes_losses = GetLostDistributionArr(aperture_nodes, params_dict["lostbunch"])
total_loss = sum([loss for (node, loss) in aperture_nodes_losses])
if _mpi_rank == 0:
    filename = get_filename("losses.txt")
    print("Total loss = {:.2e}".format(total_loss))
    print("Saving loss vs. node array to {}".format(filename))
    file = open(filename, "w")
    file.write("node position loss\n")
    for (node, loss) in aperture_nodes_losses:
        file.write("{} {} {}\n".format(node.getName(), node.getPosition(), loss))
    file.close()

if _mpi_rank == 0:
    print("timestamp = {}".format(timestamp))