from __future__ import print_function
import os
import sys

import numpy as np
from tqdm import trange
from tqdm import tqdm

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
from orbit.utils import consts
import orbit_mpi
from orbit_mpi import mpi_comm
from orbit_mpi import mpi_datatype
from orbit_mpi import mpi_op
from orbit_utils import Matrix


def initialize(mass=None, kin_energy=None):
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


def get_coords(bunch):
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


def get_coords_global(bunch):
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
    _mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)
    return


def set_coords(bunch, X, start_index=0):
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


def set_current(bunch, current=None, freq=None):
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


def center(bunch):
    """Shift the bunch so that first-order moments are zero."""
    bunch_twiss_analysis = BunchTwissAnalysis()
    bunch_twiss_analysis.analyzeBunch(bunch)
    centroid = [bunch_twiss_analysis.getAverage(i) for i in range(6)]
    return shift(bunch, centroid)


def decorrelate_x_y_z(bunch, verbose=False):
    """Decorrelate x-y-z.
    
    How should this work with MPI?
    """
    if verbose:
        print('Decorrelating x-xp, y-yp, z-dE...')
    X = get_coords(bunch)
    for i in range(0, X.shape[1], 2):
        idx = np.random.permutation(np.arange(X.shape[0]))
        X[:, i : i + 2] = X[idx, i : i + 2]
    bunch = set_coords(bunch, X)
    if verbose:
        print('Decorrelation complete.')
    return bunch


def downsample(bunch, samples=1, verbose=False):
    if verbose:
        print('Downsampling bunch (samples={})...'.format(samples))
    X = get_coords(bunch)

    if 0 < samples < 1:
        samples = samples * X.shape[0]
    samples = int(np.clip(samples, 1, X.shape[0]))
    idx = np.random.choice(X.shape[0], samples, replace=False)
    X = X[idx, :]

    new_bunch = Bunch()
    bunch.copyEmptyBunchTo(new_bunch)
    new_bunch = set_coords(new_bunch, X)
    new_bunch.macroSize(bunch.macroSize() * (bunch.getSize() / new_bunch.getSize()))
    new_bunch.copyBunchTo(bunch)
    if verbose:
        print('Downsampling complete.')
    return bunch


def reverse(bunch):
    """Reverse the bunch propagation direction.

    Since the tail becomes the head of the bunch, the sign of z
    changes but the sign of dE does not change.
    """
    for i in range(bunch.getSize()):
        bunch.xp(i, -bunch.xp(i))
        bunch.yp(i, -bunch.yp(i))
        bunch.z(i, -bunch.z(i))
    return bunch


def shift(bunch, location=None, verbose=False):
    """Shift the bunch centroid in phase space."""
    x, xp, y, yp, z, dE = location
    if verbose:
        print(
            "Shifting bunch centroid...",
            "  delta_x = {:.3f}".format(x),
            "  delta_x = {:.3f}".format(y),
            "  delta_z = {:.3f}".format(z),
            "  delta_xp = {:.3f}".format(xp),
            "  delta_yp = {:.3f}".format(yp),
            "  delta_dE = {:.3f}".format(dE),
        )
    for i in range(bunch.getSize()):
        bunch.x(i, bunch.x(i) + x)
        bunch.y(i, bunch.y(i) + y)
        bunch.z(i, bunch.z(i) + z)
        bunch.xp(i, bunch.xp(i) + xp)
        bunch.yp(i, bunch.yp(i) + yp)
        bunch.dE(i, bunch.dE(i) + dE)
    return bunch


def load(
    filename=None,
    file_format='pyorbit',
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
        print("Loading bunch from file '{}'...".format(filename))
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
            "Bunch loaded (nparts={}, macrosize={}).".format(
                bunch.getSize(),
                bunch.macroSize(),
            )
        )
    return bunch


def generate(dist=None, n_parts=0, verbose=0, bunch=None):
    """Generate bunch from distribution generator. (MPI compatible.)
    
    Parameters
    ----------
    dist : orbit.bunch_generators.distribution_generators
        Must have method `getCoordinates()` that returns (x, xp, y, yp, z, dE).
    n_parts : int
        The number of particles to generate.
    verbose : bool
        Whether to use progess bar when filling bunch.
    bunch : Bunch
        If provided, it is repopulated with `dist_gen`.
    """
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
    _mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)
    data_type = mpi_datatype.MPI_DOUBLE
    main_rank = 0
    
    if bunch is None:
        bunch = Bunch()
    else:
        bunch.deleteAllParticles()
        
    _range = range(n_parts)
    if verbose:
        _range = tqdm(_range)
    for i in _range:
        (x, xp, y, yp, z, dE) = dist.getCoordinates()
        (x, xp, y, yp, z, dE) = orbit_mpi.MPI_Bcast(
            (x, xp, y, yp, z, dE),
            mpi_datatype.MPI_DOUBLE,
            main_rank,
            _mpi_comm,
        )
        if i % _mpi_size == _mpi_rank:
            bunch.addParticle(x, xp, y, yp, z, dE)
    return bunch    


def slice_planar(bunch):
    raise NotImplementedError