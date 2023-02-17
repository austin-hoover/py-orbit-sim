from __future__ import print_function
import os
import sys
import time
from pathlib import Path

from bunch import Bunch
from orbit.lattice import AccActionsContainer
from orbit.py_linac.lattice import LinacAccLattice
from orbit.py_linac.lattice import LinacAccNodes
from orbit.utils import consts
import orbit_mpi

from btf_lattice import BTFLatticeGenerator

sys.path.append(os.getcwd())
from pyorbit_sim.linac import Monitor
from pyorbit_sim.linac import track_bunch


# Setup
# ------------------------------------------------------------------------------

_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

outdir = "./data/"
script_name = Path(__file__).stem

# I think all processess should return the same timestamp, but to be safe:
timestamp = ''
if _mpi_rank == 0:
    timestamp = time.strftime("%y%m%d%H%M%S")
timestamp = orbit_mpi.MPI_Bcast(timestamp, orbit_mpi.mpi_datatype.MPI_CHAR, 0, _mpi_comm)
prefix = "{}-{}".format(timestamp, script_name)
    
    
# Lattice
# ------------------------------------------------------------------------------
max_drift_length = 0.010  # [m]
btf_lattice_generator = BTFLatticeGenerator(
    coef_filename="/home/46h/repo/btf-lattice/magnets/default_i2gl_coeff.csv",
)
btf_lattice_generator.init_lattice(
    xml="/home/46h/repo/btf-lattice/xml/btf_lattice_default.xml",
    beamlines=["MEBT1", "MEBT2", "MEBT3"], 
    max_drift_length=max_drift_length,
)
btf_lattice_generator.update_quads_from_mstate(
    "/home/46h/repo/btf-lattice/mstate/TransmissionBS34_04212022.mstate",
    value_type="current",
)
btf_lattice_generator.make_pmq_fields_overlap(z_step=max_drift_length, verbose=True)
btf_lattice_generator.add_aperture_nodes(drift_step=0.1)
btf_lattice_generator.add_space_charge_nodes(
    grid_size_x=64, 
    grid_size_y=64,
    grid_size_z=64,
    path_length_min=max_drift_length,
    n_bunches=3,
    freq=402.5e6,
)
lattice = btf_lattice_generator.lattice

# Save node positions.
if _mpi_rank == 0:
    file = open(os.path.join(outdir, "{}_nodes.dat".format(prefix)), "w")
    file.write("node, position\n")
    for node in lattice.getNodes():
        file.write("{}, {}, {}\n".format(node.getName(), node.getPosition(), node.getLength()))
    file.close()

    # Write lattice structure to file.
    file = open(os.path.join(outdir, "{}_lattice_structure.txt".format(prefix)), "w")
    file.write(lattice.structureToText())
    file.close()


# Bunch
# ------------------------------------------------------------------------------

filename = os.path.join(
    "/home/46h/BTF/meas_analysis/2022-06-26_scan-xxpy-image-ypdE/data/",
    "220626140058-scan-xxpy-image-ypdE_samp6D_1.00e+06_decorrelated.dat",
)
bunch = Bunch()
if _mpi_rank == 0:
    print("Generating bunch from file '{}'.".format(filename))
bunch.readBunch(filename)
bunch.mass(0.939294)  # [GeV / c^2]
bunch.charge(-1.0)  # [elementary charge units]
bunch.getSyncParticle().kinEnergy(0.0025)  # [GeV]
bunch_current = 0.050  # [A]
bunch_freq = 402.5e6  # [Hz]
bunch_charge = bunch_current / bunch_freq
intensity = bunch_charge / abs(float(bunch.charge()) * consts.charge_electron)
bunch_size_global = bunch.getSizeGlobal()
bunch.macroSize(intensity / bunch_size_global)

if _mpi_rank == 0:
    print("Bunch parameters:")
    print("  charge = {}".format(bunch.charge()))
    print("  mass = {} [GeV / c^2]".format(bunch.mass()))
    print("  kinetic energy = {} [GeV]".format(bunch.getSyncParticle().kinEnergy()))
    print("  macrosize = {}".format(bunch.macroSize()))
    print("  size (local) = {:.2e}".format(bunch.getSize()))
    print("  size (global) = {:.2e}".format(bunch_size_global))
    
    
# Sim
# ------------------------------------------------------------------------------

start = "MEBT:HZ04"  # start node (name or position)
stop = "MEBT:HZ34a"  # stop node (name or position)

monitor = Monitor(
    start_position=0.0, # this will be set automatically in `track_bunch`.
    plotter=None,
    verbose=True,
    track_history=True,
    dispersion_flag=True,
    emit_norm_flag=False,
)

track_bunch(bunch, lattice, monitor=monitor, start=start, stop=stop, verbose=True)

if _mpi_rank == 0 and monitor.track_history:
    print("Writing RMS history to file.")
    monitor.write(filename=os.path.join(outdir, "{}_history.dat".format(prefix)), delimiter=",")

if _mpi_rank == 0:
    print("Saving output bunch to file.")
bunch.dumpBunch(os.path.join(outdir, "{}_bunch_{}.dat".format(prefix, stop)))