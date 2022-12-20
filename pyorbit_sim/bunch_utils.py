from __future__ import print_function

import numpy as np
import psdist.bunch as psb

from bunch import Bunch
from bunch import BunchTwissAnalysis
from orbit.bunch_generators import GaussDist1D
from orbit.bunch_generators import GaussDist2D
from orbit.bunch_generators import GaussDist3D
from orbit.bunch_generators import KVDist1D
from orbit.bunch_generators import KVDist2D
from orbit.bunch_generators import KVDist3D
from orbit.bunch_generators import TwissAnalysis
from orbit.bunch_generators import TwissContainer
from orbit.bunch_generators import WaterBagDist1D
from orbit.bunch_generators import WaterBagDist2D
from orbit.bunch_generators import WaterBagDist3D
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.teapot import teapot
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.twiss import twiss
from orbit.utils import consts
import orbit_mpi
from orbit_mpi import mpi_comm
from orbit_mpi import mpi_datatype
from orbit_mpi import mpi_op
from orbit_utils import Matrix


def initialize_bunch(mass=None, kin_energy=None):
    """Create and initialize Bunch.

    Parameters
    ----------
    mass, kin_energy : float
        Mass [GeV/c^2] and kinetic energy [GeV] per bunch particle.

    Returns
    -------
    bunch : Bunch
        A Bunch object with the given mass and kinetic energy.
    params_dict : dict
        Dictionary with reference to Bunch.
    """
    bunch = Bunch()
    bunch.mass(mass)
    bunch.getSyncParticle().kinEnergy(kin_energy)
    params_dict = {"bunch": bunch}
    return bunch, params_dict


def get_bunch_coords(bunch):
    """Return phase space coordinate array from bunch."""
    X = np.zeros((bunch.getSize(), 6))
    for i in range(bunch.getSize()):
        X[i, 0] = bunch.x(i)
        X[i, 1] = bunch.xp(i)
        X[i, 2] = bunch.y(i)
        X[i, 3] = bunch.yp(i)
        X[i, 4] = bunch.z(i)
        X[i, 5] = bunch.dE(i)
    return X


def set_bunch_coords(bunch, X, start_index=0):
    """Assign phase space coordinate array to bunch.

    bunch : Bunch
        The bunch to modify. The bunch is resized if space needs to be made for
        the new particles. The bunch is not resized if there is already space.
    X : ndarray, shape (k, 6)
        The phase space coordinates to add (columns: x, xp, y, yp, z, dE).
    start_index : int
        The bunch is filled starting from this particle index.

    Returns
    -------
    bunch : Bunch
        The modified bunch.
    """
    overflow = (X.shape[0] + start_index) - bunch.getSize()
    if overflow > 0:
        for _ in range(overflow):
            bunch.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for i, (x, xp, y, yp, z, dE) in enumerate(X, start=start_index):
        bunch.x(i, x)
        bunch.y(i, y)
        bunch.z(i, z)
        bunch.xp(i, xp)
        bunch.yp(i, yp)
        bunch.dE(i, dE)
    return bunch


def set_bunch_current(bunch, current=None, freq=None):
    """Set macro-size from current [A] and frequency [Hz]."""
    charge_bunch = current / freq
    charge_particle = abs(float(bunch.charge()) * consts.charge_electron)
    intensity = charge_bunch / charge_particle
    macro_size = intensity / bunch.getSizeGlobal()
    bunch.macroSize(macro_size)
    return bunch


def get_z_to_phase_coeff(bunch, freq=None):
    """Return coefficient to calculate phase [degrees] from z [m]."""
    wavelength = consts.speed_of_light / freq
    return -360.0 / (bunch.getSyncParticle().beta() * wavelength)


def shift_bunch(bunch, new_center=None):
    """Shift the bunch centroid in phase space."""
    x, xp, y, yp, z, dE = new_center
    for i in range(bunch.getSize()):
        bunch.x(i, bunch.x(i) + x)
        bunch.y(i, bunch.y(i) + y)
        bunch.z(i, bunch.z(i) + z)
        bunch.xp(i, bunch.xp(i) + xp)
        bunch.yp(i, bunch.yp(i) + yp)
        bunch.dE(i, bunch.dE(i) + dE)
    return bunch


def center_bunch(bunch):
    """Shift the bunch so that first-order moments are zero."""
    twiss = BunchTwissAnalysis()
    twiss.analyzeBunch(bunch)
    return shift_bunch(bunch, *[twiss.getAverage(i) for i in range(6)])


def reverse_bunch(bunch):
    """Reverse the bunch propagation direction.

    Since the tail becomes the head of the bunch, the sign of z
    changes but the sign of dE does not change.
    """
    for i in range(bunch.getSize()):
        bunch.xp(i, -bunch.xp(i))
        bunch.yp(i, -bunch.yp(i))
        bunch.z(i, -bunch.z(i))
        


def load_bunch(
    filename=None,
    file_format="pyorbit",
    bunch=None,
    verbose=False,
):
    """Load bunch from coordinates file.

    Parameters
    ----------
    filename : str
        Path the file.
    file_format : str
        'pyorbit':
            The expected header format is:
        'parmteq':
            The expected header format is:
                Number of particles    =
                Beam current           =
                RF Frequency           =
                The input file particle coordinates were written in double precision.
                x(cm)             xpr(=dx/ds)       y(cm)             ypr(=dy/ds)       phi(radian)        W(MeV)
    verbose : bool
        Whether to print intro/exit messages.
    bunch : Bunch
        If None, create a new bunch; otherwise, load into this bunch.
    """
    if verbose:
        print("Reading bunch from file '{}'...".format(filename))
    if not os.path.isfile(filename):
        raise ValueError("File '{}' does not exist.".format(filename))
    if bunch is None:
        bunch = Bunch()
    if file_format == "pyorbit":
        bunch.readBunch(filename)
    elif file_format == "parmteq":
        # Read data.
        header = np.genfromtxt(filename, max_rows=3, usecols=[0, 1, 2, 3, 4], dtype=str)
        n_parts = int(header[0, 4])
        current = np.float(header[1, 3])
        freq = np.float(header[2, 3]) * 1e6  # MHz to Hz
        data = np.loadtxt(filename, skiprows=5)

        # Trim off-energy particles.
        kin_energy = np.mean(data[:, 5])  # center energy [MeV]
        ind = np.where(np.abs(data[:, 5] - kin_energy) < (0.05 * kin_energy))[0]
        n_parts = len(ind)
        bunch.getSyncParticle().kinEnergy(kin_energy * 1e-3)

        # Unit conversion.
        dE = (data[ind, 5] - kin_energy) * 1e-3  # MeV to GeV
        x = data[ind, 0] * 1e-2  # cm to m
        xp = data[ind, 1]  # radians
        y = data[ind, 2] * 1e-2  # cm to m
        yp = data[ind, 3]  # radians
        phi = data[ind, 4]  # radians
        z = np.rad2deg(phi) / get_z_to_phase_coeff(bunch, freq=freq)

        # Add particles.
        for i in range(n_parts):
            bunch.addParticle(x[i], xp[i], y[i], yp[i], z[i], dE[i])
    else:
        raise KeyError("Unrecognized format '{}'.".format(file_format))
    if verbose:
        print(
            "Bunch loaded (nparts={} macrosize={}).".format(
                bunch.getSize(), 
                bunch.macroSize(),
            )
        )
    return bunch


def gen_bunch_from_twiss(
    n_parts=0, twiss_x=None, twiss_y=None, twiss_z=None, dist_gen=None, **dist_gen_kws
):
    """Generate bunch from Twiss parameters."""
    comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    rank = orbit_mpi.MPI_Comm_rank(comm)
    size = orbit_mpi.MPI_Comm_size(comm)
    data_type = mpi_datatype.MPI_DOUBLE
    main_rank = 0
    bunch = Bunch()
    distributor = dist_gen(twiss_x, twiss_y, twiss_z, **dist_gen_kws)
    bunch.getSyncParticle().time(0.0)
    for i in range(n_parts):
        x, xp, y, yp, z, dE = distributor.getCoordinates()
        x, xp, y, yp, z, dE = orbit_mpi.MPI_Bcast(
            (x, xp, y, yp, z, dE),
            data_type,
            main_rank,
            comm,
        )
        if i % size == rank:
            bunch.addParticle(x, xp, y, yp, z, dE)
    return bunch


def decorrelate_bunch(bunch):
    X = get_bunch_coords(bunch)
    X = psb.decorrelate(X)
    bunch = set_bunch_coords(bunch, X)
    return bunch


def downsample_bunch(bunch, samples=None):
    X = get_bunch_coords(bunch)
    X = psb.downsample(X, samples)
    bunch = set_bunch_coords(bunch, X)