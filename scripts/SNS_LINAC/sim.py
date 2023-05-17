"""SNS Linac simulation."""
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


# Setup
# --------------------------------------------------------------------------------------

save = True  # no output if False

# MPI
_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

# Create output directory and save script info.
man = ScriptManager(datadir="/home/46h/sim_data/", path=pathlib.Path(__file__))
if save:
    man.save_info()
    man.save_script_copy()
if _mpi_rank == 0:
    print("Script info:")
    print("save = {}".format(save))
    pprint(man.get_info())

file_path = os.path.dirname(os.path.realpath(__file__))


# Lattice
# --------------------------------------------------------------------------------------

# Generate SNS linac lattice from XML file.
xml_file_name = os.path.join(file_path, "data/sns_linac.xml")
max_drift_length = 0.01  # [m]
sequences = [
    "MEBT",
    "DTL1",
    "DTL2",
    "DTL3",
    "DTL4",
    "DTL5",
    "DTL6",
    "CCL1",
    "CCL2",
    "CCL3",
    "CCL4",
    "SCLMed",
    "SCLHigh",
    "HEBT1",
    "HEBT2",
]
sns_linac_factory = SNS_LinacLatticeFactory()
sns_linac_factory.setMaxDriftLength(max_drift_length)
lattice = sns_linac_factory.getLinacAccLattice(sequences, xml_file_name)
if _mpi_rank == 0:
    print("Initialized lattice.")
    print("XML filename = {}".format(xml_file_name))
    print("lattice length = {:.3f} [m])".format(lattice.getLength()))


# Save lattice structure.
if save and _mpi_rank == 0:
    # Node start/stop positions.
    file = open(man.get_filename("lattice_nodes.txt"), "w")
    file.write("node position length\n")
    for node in lattice.getNodes():
        file.write(
            "{} {} {}\n".format(node.getName(), node.getPosition(), node.getLength())
        )
    file.close()
    # Full lattice structure.
    file = open(man.get_filename("lattice_structure.txt"), "w")
    file.write(lattice.structureToText())
    file.close()

    
# Set the RF gap model.
#   BaseRfGap / BaseRfGap_slow: uses only E0TL * cos(phi) * J0(kr), with E0TL = const.
#   MatrixRfGap / MatrixRfGap_slow: uses a matrix approach like envelope codes.
#   RfGapTTF / RfGapTTF_slow: uses Transit Time Factors (TTF) like PARMILA.
rf_gap_model = RfGapTTF
for rf_gap in lattice.getRF_Gaps():
    rf_gap.setCppGapModel(rf_gap_model())

    
# Set overlapping RF and quad fields.
fields_filename = os.path.join(file_path, "data/sns_rf_fields.xml")
z_step = 0.002
        
## Only RF gaps will be replaced with non-zero length models. Quads stay hard-edged. 
## Such approach will not work for DTL cavities - RF and quad fields are overlapped 
## for DTL.
# seq_names = ["MEBT", "CCL1", "CCL2", "CCL3", "CCL4", "SCLMed"]
# Replace_BaseRF_Gap_to_AxisField_Nodes(lattice, z_step, fields_filename, seq_names)

## Hard-edge quad models will be replaced with soft-edge models. It is possible for DTL 
## also - if the RF gap models are zero-length ones. 
#
# seq_names = ["MEBT", "DTL1", "DTL2", "DTL3", "DTL4", "DTL5", "DTL6"]
# Replace_Quads_to_OverlappingQuads_Nodes(lattice, z_step, seq_names, [], SNS_EngeFunctionFactory)

## Hard-edge quad and zero-length RF gap models will be replaced with soft-edge quads
## and field-on-axis RF gap models. Can be used for any sequences, no limitations.
#
# seq_names = ["MEBT", "DTL1", "DTL2", "DTL3", "DTL4", "DTL5", "DTL6"]
# Replace_BaseRF_Gap_and_Quads_to_Overlapping_Nodes(
#     lattice, z_step, fields_filename, seq_names, [], SNS_EngeFunctionFactory
# )


# Add space charge nodes.
sc_solver = "3D"  # {"3D", "ellipsoid", None}
sc_grid_size = (64, 64, 64)
sc_path_length_min = 0.01
sc_n_bunches = 1
sc_freq = 402.5e6
if sc_solver == "3D":
    sc_calc = SpaceChargeCalc3D(sc_grid_size[0], sc_grid_size[1], sc_grid_size[2])
    if sc_n_bunches > 1: 
        sc_calc.numExtBunches(n_bunches)
        sc_calc.freqOfBunches(freq)
    sc_nodes = setSC3DAccNodes(lattice, sc_path_length_min, sc_calc)
elif sc_solver == "ellipsoid":
    n_ellipsoids = 1
    sc_calc = SpaceChargeCalcUnifEllipse(n_ellipsoids)
    sc_nodes = setUniformEllipsesSCAccNodes(lattice, sc_path_length_min, sc_calc)
if sc_solver is not None and _mpi_rank == 0:
    sc_lengths = [sc_node.getLengthOfSC() for sc_node in sc_nodes]
    min_sc_length = min(min(sc_lengths), lattice.getLength())
    max_sc_length = max(max(sc_lengths), 0.0)
    print("Added {} space charge nodes".format(len(sc_nodes)))
    print("max length = {}".format(max_sc_length))
    print("min length = {}".format(min_sc_length))


# Add aperture nodes.
x_size = 0.042
y_size = 0.042
aperture_nodes = Add_quad_apertures_to_lattice(lattice)
aperture_nodes = Add_rfgap_apertures_to_lattice(lattice, aperture_nodes)
aperture_nodes = AddMEBTChopperPlatesAperturesToSNS_Lattice(lattice, aperture_nodes)
aperture_nodes = AddScrapersAperturesToLattice(lattice, "MEBT_Diag:H_SCRP", x_size, y_size, aperture_nodes)
aperture_nodes = AddScrapersAperturesToLattice(lattice, "MEBT_Diag:V_SCRP", x_size, y_size, aperture_nodes)
if _mpi_rank == 0:
    print("Added {} aperture nodes.".format(len(aperture_nodes)))

    
# Use linac-style quads and drifts instead of TEAPOT style. (Useful when 
# the energy spread is large, but is slower and is not symplectic.)
lattice.setLinacTracker(False)


# Bunch
# --------------------------------------------------------------------------------------

# Initialize the bunch.
bunch = Bunch()
bunch.mass(0.939294)  # [GeV / c^2]
bunch.charge(-1.0)  # [elementary charge units]
bunch.getSyncParticle().kinEnergy(0.0025)  # [GeV]
current = 0.042  # [C/s]
frequency = 402.5e6  # [1/s]
intensity = (current / frequency) / abs(float(bunch.charge()) * consts.charge_electron)
gamma = bunch.getSyncParticle().gamma()
beta = bunch.getSyncParticle().beta()

# Load the bunch coordinates.
# bunch_filename = None
bunch_filename = "/home/46h/projects/BTF/sim/data/RFQ_output_PARMTEQ_50mA_42mA_1.00e+06.dat"
if bunch_filename is None:
    if _mpi_rank == 0:
        print("Generating bunch from Twiss parameters.")    
    alpha_x = -1.9620
    alpha_y = 1.7681
    alpha_z = -0.0196
    beta_x = 0.1831
    beta_y = 0.1620
    beta_z = 0.5844
    eps_x = 0.21e-6 / (beta * gamma)  # [m * rad]
    eps_y = 0.21e-6 / (beta * gamma)  # [m * rad]
    eps_z = 0.24153e-6 / (beta * gamma**3)  # [m * rad]
    eps_z = eps_z * gamma**3 * beta**2 * bunch.mass()  # [m * GeV]
    beta_z = beta_z / (gamma**3 * beta**2 * bunch.mass())
    bunch = pyorbit_sim.bunch_utils.generate(
        dist=WaterBagDist3D(
            twissX=TwissContainer(alpha_x, beta_x, eps_x),
            twissY=TwissContainer(alpha_y, beta_y, eps_y),
            twissZ=TwissContainer(alpha_z, beta_z, eps_z),
        ),
        bunch=bunch,
        n_parts=int(1e5), 
        verbose=True,
    )
else:
    if _mpi_rank == 0:
        print("Generating bunch from file '{}'.".format(bunch_filename))
    bunch.readBunch(bunch_filename)
        
# Downsample by random selection. 
## Here we assume the particles were randomly generated to begin with, so we
## just use the first k indices. Note that random selection is not guaranteed
## to preserve the 6D phase space distribution.
##
## Need to think more about how to work with MPI.
samples = None
if samples is not None and bunch_size_global > samples:
    new_bunch = Bunch()
    bunch.copyEmptyBunchTo(new_bunch)
    print(new_bunch.getSize())
    for i in range(samples):
        new_bunch.addParticle(
            bunch.x(i),
            bunch.xp(i), 
            bunch.y(i), 
            bunch.yp(i), 
            bunch.z(i), 
            bunch.dE(i),
        )
    new_bunch.copyBunchTo(bunch)
    bunch.macroSize(intensity / samples)

# Decorrelate x-y-z.
decorrelate = False
if decorrelate:
    bunch = pyorbit_sim.bunch_utils.decorrelate_x_y_z(bunch, verbose=True)

# If `dist` is not None, generate an RMS-equivalent distribution in x-x', y-y', and z-z' 
# using an analytic distribution function (Gaussian, KV, Waterbag). Reconstruct the the 
# six-dimensional distribution as f(x, x', y, y', z, z') = f(x, x') f(y, y') f(z, z').
dist = GaussDist3D
n_parts = int(1e5)
if dist is not None:
    if _mpi_rank == 0:
        print("Repopulating bunch using 2D Twiss parameters and {} generator.".format(dist))
    bunch_twiss_analysis = BunchTwissAnalysis()
    order = 2
    dispersion_flag = 0
    emit_norm_flag = 0
    bunch_twiss_analysis.computeBunchMoments(bunch, order, dispersion_flag, emit_norm_flag)    
    eps_x = bunch_twiss_analysis.getEffectiveEmittance(0)
    eps_y = bunch_twiss_analysis.getEffectiveEmittance(1)
    eps_z = bunch_twiss_analysis.getEffectiveEmittance(2)
    beta_x = bunch_twiss_analysis.getEffectiveBeta(0)
    beta_y = bunch_twiss_analysis.getEffectiveBeta(1)
    beta_z = bunch_twiss_analysis.getEffectiveBeta(2)
    alpha_x = bunch_twiss_analysis.getEffectiveAlpha(0)
    alpha_y = bunch_twiss_analysis.getEffectiveAlpha(1)
    alpha_z = bunch_twiss_analysis.getEffectiveAlpha(2)    
    dist = dist(
        twissX=TwissContainer(alpha_x, beta_x, eps_x),
        twissY=TwissContainer(alpha_y, beta_y, eps_y),
        twissZ=TwissContainer(alpha_z, beta_z, eps_z),
    )
    bunch = pyorbit_sim.bunch_utils.generate(dist=dist, n_parts=n_parts, bunch=bunch, verbose=True)
    
# Set the macrosize.
bunch_size_global = bunch.getSizeGlobal()
macro_size = intensity / bunch_size_global
bunch.macroSize(macro_size)
    
# Print bunch parameters.
twiss_analysis = BunchTwissAnalysis()
twiss_analysis.analyzeBunch(bunch)
if _mpi_rank == 0:
    print("Bunch parameters:")
    print("  charge = {}".format(bunch.charge()))
    print("  mass = {} [GeV / c^2]".format(bunch.mass()))
    print("  kinetic energy = {} [GeV]".format(bunch.getSyncParticle().kinEnergy()))
    print("  macrosize = {}".format(bunch.macroSize()))
    print("  size (local) = {:.2e}".format(bunch.getSize()))
    print("  size (global) = {:.2e}".format(bunch_size_global))    
    print("Twiss parameters:")
    print("  alpha_x = {}".format(twiss_analysis.getTwiss(0)[0]))
    print("  alpha_y = {}".format(twiss_analysis.getTwiss(1)[0]))
    print("  alpha_z = {}".format(twiss_analysis.getTwiss(2)[0]))
    print("  beta_x = {}".format(twiss_analysis.getTwiss(0)[1]))
    print("  beta_y = {}".format(twiss_analysis.getTwiss(1)[1]))
    print("  beta_z = {}".format(twiss_analysis.getTwiss(2)[1]))
    print("  eps_x = {} [mm * mrad]".format(1.0e6 * twiss_analysis.getTwiss(0)[3]))
    print("  eps_y = {} [mm * mrad]".format(1.0e6 * twiss_analysis.getTwiss(1)[3]))
    print("  eps_z = {} [mm * MeV]".format(1.0e6 * twiss_analysis.getTwiss(2)[3]))

    
## Assign ID number to each particle.
# ParticleIdNumber.addParticleIdNumbers(bunch)
# copyCoordsToInitCoordsAttr(bunch)


# Tracking
# --------------------------------------------------------------------------------------

start = 0  # start node (name or position)
stop = 30.0  # stop node (name or position)s
save_input_bunch = True
save_output_bunch = True


# Bunch coordinate writer
writer = pyorbit_sim.linac.BunchWriter(
    folder=man.outdir, 
    prefix=man.prefix, 
    index=1, 
)
    
    
def transform(X):
    """Normalize the 2D phase spaces."""
    Sigma = np.cov(X.T)
    Xn = np.zeros(X.shape)
    for i in range(0, X.shape[1], 2):
        sigma = Sigma[i : i + 2, i : i + 2]
        eps = np.sqrt(np.linalg.det(sigma))
        alpha = -sigma[0, 1] / eps
        beta = sigma[0, 0] / eps
        Xn[:, i] = X[:, i] / np.sqrt(beta)
        Xn[:, i + 1] = (np.sqrt(beta) * X[:, i + 1]) + (alpha * X[:, i] / np.sqrt(beta))
        Xn[:, i : i + 2] = Xn[:, i : i + 2] / np.sqrt(eps)
    return Xn
    
plotter = pyorbit_sim.plotting.Plotter(
    transform=transform, 
    folder=man.outdir,
    default_save_kws=None, 
)
plotter.add_function(
    pyorbit_sim.plotting.proj2d_three_column, 
    save_kws=None, 
    name=None, 
    bins=32,
)


monitor = pyorbit_sim.linac.Monitor(
    position_offset=0.0,  # will be set automatically in `pyorbit_sim.linac.track`.
    stride={
        "update": 0.1,  # [m]
        "write_bunch": (5.0 if save else None),  # [m]
        "plot_bunch": (None if save else None),  # [m]
    },
    writer=writer,
    plotter=None,
    track_history=True,
    track_rms=True,
    dispersion_flag=False,
    emit_norm_flag=False,
    verbose=True,
)


# Record synchronous particle time of arrival at each accelerating cavity.
if _mpi_rank == 0:
    print("Tracking design bunch...")
lattice.trackDesignBunch(bunch)
if _mpi_rank == 0:
    print("Design bunch tracking complete.")
    
# Save input bunch.
if save and save_input_bunch:
    if start is None or type(start) is not str:
        filename = man.get_filename("bunch_0_START.dat")
    else:
        filename = man.get_filename("bunch_0_{}.dat".format(start))
    if _mpi_rank == 0:
        print("Saving bunch to file {}".format(filename))
    bunch.dumpBunch(filename)    
    
# Track
if _mpi_rank == 0:
    print("Tracking...")
        
params_dict = pyorbit_sim.linac.track(
    bunch, 
    lattice, 
    monitor=monitor, 
    start=start, 
    stop=stop, 
    verbose=True
)

# Save history array.
if _mpi_rank == 0 and monitor.track_history and save:
    filename = man.get_filename("history.dat")
    print("Writing history to {}".format(filename))
    monitor.write_history(filename, delimiter=",")
    
    
# Save lost particles.
lostbunch = params_dict["lostbunch"]
aprt_nodes_losses = GetLostDistributionArr(aperture_nodes, lostbunch)
total_loss = 0.0

for (node, loss) in aprt_nodes_losses:
    if _mpi_rank == 0:
        print(
            "node={:30s},".format(node.getName()), 
            "pos={:9.3f},".format(node.getPosition()), 
            "loss= {:6.0f}".format(loss),
        )
    total_loss += loss

if _mpi_rank == 0:
    print("Total loss = {:.2e}".format(total_loss))
    
if save:
    filename = man.get_filename("losses.txt")
    if _mpi_rank == 0:
        print("Saving loss vs. node array to {}".format(filename))
    file = open(filename, "w")
    for (node, loss) in aprt_nodes_losses:
        file.write("{} {}\n".format(node.getName(), loss))
    file.close()

    filename = man.get_filename("bunch_lost.dat".format(stop))
    if _mpi_rank == 0:
        print("Saving lost bunch to file {}".format(filename))
    bunch.dumpBunch(filename)    
    
    
# Save output bunch.
if save and save_output_bunch:
    if stop is None or type(stop) is not str:
        filename = man.get_filename("bunch_{}_STOP.dat".format(writer.index))
    else:
        filename = man.get_filename("bunch_{}_{}.dat".format(writer.index, stop))
    if _mpi_rank == 0:
        print("Saving bunch to file {}".format(filename))
    bunch.dumpBunch(filename)

if _mpi_rank == 0:
    print("timestamp = {}".format(man.timestamp))