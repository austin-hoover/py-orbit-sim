from __future__ import print_function
import math
import time

from bunch import Bunch
from bunch import BunchTwissAnalysis
from orbit.bunch_generators import GaussDist3D
from orbit.bunch_generators import TwissContainer
from orbit.lattice import AccActionsContainer
from orbit.space_charge.sc3d import setSC3DAccNodes
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import DriftTEAPOT
from orbit.utils import consts
import orbit_mpi
from spacecharge import SpaceChargeCalc3D


# Setup MPI
_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)
if _mpi_rank == 0:
    print("mpi_comm = {}".format(_mpi_comm))
    print("mpi size = {}".format(_mpi_size))


# Create the bunch. The bunch knows about MPI.
bunch = Bunch()
bunch.mass(consts.mass_proton)
bunch.getSyncParticle().kinEnergy(0.0025)
params_dict = {"bunch": bunch}

# Add particles to the bunch on this MPI node (total number of particles
# divided by number of processes).
n_parts_total = int(1e6)
alpha_x = 0.0
alpha_y = 0.0
alpha_z = 0.0
beta_x = 1.0
beta_y = 1.0
beta_z = 0.2e3
eps_x = 1.0e-6
eps_y = 1.0e-6
eps_z = 5.0e-9
dist = GaussDist3D(
    twissX=TwissContainer(alpha_x, beta_x, eps_x),
    twissY=TwissContainer(alpha_y, beta_y, eps_y),
    twissZ=TwissContainer(alpha_z, beta_z, eps_z)
)
for _ in range(int(n_parts_total / _mpi_size)):
    x, xp, y, yp, z, dE = dist.getCoordinates()
    bunch.addParticle(x, xp, y, yp, z, dE)

# Compute the bunch size.
bunch_size_global = bunch.getSizeGlobal()
print("(rank {}) bunch.getSize() = {}".format(_mpi_rank, bunch.getSize()))
if _mpi_rank == 0:
    print("bunch.getSizeGlobal() = {}".format(bunch_size_global))
    
# Set the bunch macrosize.
freq = 402.5e6  # [Hz]
current = 0.050  # [A]
charge = current / freq
intensity = charge / (abs(bunch.charge()) * consts.charge_electron)
macro_size = intensity / bunch_size_global
bunch.macroSize(macro_size)


# Track through a drift with 3D FFT space charge nodes.

class Monitor:
    def __init__(self, rms=False):
        self.rms = rms
        self.start_time = None
        self.position = 0.0

    def action(self, params_dict):
        position = params_dict["path_length"]
        if position <= self.position:
            return
        self.position = position
        
        if self.start_time is None:
            self.start_time = time.clock()
        time_ellapsed = time.clock() - self.start_time
        
        bunch = params_dict["bunch"]
        bunch_twiss_analysis = BunchTwissAnalysis()
        order = 2
        dispersion_flag = 0
        emit_norm_flag = 0
        bunch_twiss_analysis.computeBunchMoments(bunch, order, dispersion_flag, emit_norm_flag)
        xrms = 1000.0 * math.sqrt(bunch_twiss_analysis.getCorrelation(0, 0))
        yrms = 1000.0 * math.sqrt(bunch_twiss_analysis.getCorrelation(2, 2))  
        zrms = 1000.0 * math.sqrt(bunch_twiss_analysis.getCorrelation(4, 4))  
        if _mpi_rank == 0:
            print(
                "time={:.3f}, s={:.3f}, xrms={:.3f}, yrms={:.3f}, zrms={:.3f}"
                .format(time_ellapsed, position, xrms, yrms, zrms)
            )

                
distance = 0.600  # [m]
delta_s = 0.005  # [m]
lattice = TEAPOT_Lattice()
for _ in range(int(distance / delta_s)):
    node = DriftTEAPOT()
    node.setLength(delta_s)
    lattice.addNode(node)
lattice.initialize()

grid_size_x = 64
grid_size_y = 64
grid_size_z = 64
path_length_min = delta_s
sc_calc = SpaceChargeCalc3D(grid_size_x, grid_size_y, grid_size_z)
sc_nodes = setSC3DAccNodes(lattice, path_length_min, sc_calc)
        
monitor = Monitor(rms=True)
action_container = AccActionsContainer()
action_container.addAction(monitor.action, AccActionsContainer.EXIT)
lattice.trackBunch(bunch, params_dict, actionContainer=action_container)
bunch.dumpBunch("bunch_out.dat")