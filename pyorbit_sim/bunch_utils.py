from __future__ import print_function
import collections
import math
import os
from pprint import pprint
from pprint import pprint
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

import pyorbit_sim


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
    raise NotImplementedError


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


def linear_transform(bunch, matrix):
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.shape[0] == 6
    for i in range(bunch.getSize()):
        x = bunch.x(i)
        y = bunch.y(i)
        z = bunch.z(i)
        xp = bunch.xp(i)
        yp = bunch.yp(i)
        dE = bunch.dE(i)
        (x, xp, y, yp, z, dE) = np.matmul(matrix, (x, xp, y, yp, z, dE))
        bunch.x(i, x)
        bunch.y(i, y)
        bunch.z(i, z)
        bunch.xp(i, xp)
        bunch.yp(i, yp)
        bunch.dE(i, dE)
    return bunch
    

def get_intensity(current=None, frequency=None, charge=-1.0):
    return (current / frequency) / (abs(charge) * consts.charge_electron)


def set_current(bunch, current=None, frequency=None):
    """Set macro-size from current [A] and frequency [Hz]."""
    intensity = get_intensity(current, frequency)
    macro_size = intensity / bunch.getSizeGlobal()
    bunch.macroSize(macro_size)
    return bunch


def get_z_to_phase_coeff(bunch, frequency=None):
    """Return coefficient to calculate phase [degrees] from z [m]."""
    wavelength = consts.speed_of_light / frequency
    return -360.0 / (bunch.getSyncParticle().beta() * wavelength)


def get_z_rms_deg(bunch, frequency=None, z_rms=None):
    """Convert z rms from [m] to [deg]."""
    return -get_z_to_phase_coeff(bunch, frequency) * z_rms


def decorrelate_x_y_z(bunch, verbose=False):
    """Decorrelate x-y-z.
    
    Does not work with MPI currently.
    """
    n = bunch.getSizeGlobal()
    idx_x = np.random.permutation(np.arange(n))
    idx_y = np.random.permutation(np.arange(n))
    idx_z = np.random.permutation(np.arange(n))
    print("Building decorrelated bunch...")
    bunch_out = Bunch()
    bunch.copyEmptyBunchTo(bunch_out)
    for i, j, k in tqdm(zip(idx_x, idx_y, idx_z)):
        bunch_out.addParticle(bunch.x(i), bunch.xp(i), bunch.y(j), bunch.yp(j), bunch.z(k), bunch.dE(k))
    bunch_out.copyBunchTo(bunch)
    bunch_out.deleteAllParticles()
    return bunch


def decorrelate_xy_z(bunch, verbose=False):
    """Decorrelate xy-z.
    
    Does not work with MPI currently.
    """
    n = bunch.getSizeGlobal()
    idx_xy = np.random.permutation(np.arange(n))
    idx_z = np.random.permutation(np.arange(n))
    print("Building decorrelated bunch...")
    bunch_out = Bunch()
    bunch.copyEmptyBunchTo(bunch_out)
    for i, k in tqdm(zip(idx_xy, idx_z)):
        bunch_out.addParticle(bunch.x(i), bunch.xp(i), bunch.y(i), bunch.yp(i), bunch.z(k), bunch.dE(k))
    bunch_out.copyBunchTo(bunch)
    bunch_out.deleteAllParticles()
    return bunch


def downsample(bunch, n=1, verbose=False, method="first", conserve_intensity=True):
    """Downsample the bunch to `n` particles.
    
    Parameters
    ----------
    n : int or float
        The number of particles to keep. (The new global bunch size.)
    method : str
        "first": Keep the first n particles in the bunch. (Or keep the
        first (n / mpi_size) particles on each processor.) This method 
        assumes the particles were originally randomly generated.
    conserve_intensity : bool
        Whether to increase the macrosize after downsampling to conserve
        the bunch intensity.
    """
    if n is None:
        return bunch
    
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
    _mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)
    
    macro_size = bunch.macroSize()
    bunch_size_global = bunch.getSizeGlobal()
    intensity = macro_size * bunch_size_global
    
    n_proc = int(n / _mpi_size)
    n_proc_old = bunch.getSize()    
    if n_proc >= n_proc_old:
        return bunch    
    
    if method == "first":        
        if verbose:
            print("(rank {}) n_old={}, n_new={}".format(_mpi_rank, n_proc_old, n_proc))
        for i in reversed(range(n_proc, n_proc_old)):
            bunch.deleteParticleFast(i)
        bunch.compress()    
    else:
        raise ValueError("Invalid method")
    
    if conserve_intensity:
        bunch_size_global = bunch.getSizeGlobal()
        macro_size = intensity / bunch_size_global
        bunch.macroSize(macro_size)
    
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


def get_centroid(bunch):
    twiss_calc = BunchTwissAnalysis()
    twiss_calc.analyzeBunch(bunch)
    return np.array([twiss_calc.getAverage(i) for i in range(6)])


def shift_centroid(bunch, delta=None, verbose=False):
    """Shift the bunch centroid in phase space."""
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

    x, xp, y, yp, z, dE = delta
    if _mpi_rank == 0 and verbose:
        print("Shifting bunch centroid...")
        print(" delta x = {:.3e} [m]".format(delta[0]))
        print(" delta y = {:.3e} [m]".format(delta[2]))
        print(" delta z = {:.3e} [m]".format(delta[4]))
        print(" delta xp = {:.3e} [mrad]".format(delta[1]))
        print(" delta yp = {:.3e} [mrad]".format(delta[3]))
        print(" delta dE = {:.3e} [GeV]".format(delta[5]))
    for i in range(bunch.getSize()):
        bunch.x(i, bunch.x(i) + delta[0])
        bunch.y(i, bunch.y(i) + delta[2])
        bunch.z(i, bunch.z(i) + delta[4])
        bunch.xp(i, bunch.xp(i) + delta[1])
        bunch.yp(i, bunch.yp(i) + delta[3])
        bunch.dE(i, bunch.dE(i) + delta[5])
    if verbose:
        centroid = get_centroid(bunch)
        print("New centroid:")
        print("<x>  = {} [m]".format(centroid[0]))
        print("<xp> = {} [rad]".format(centroid[1]))
        print("<y>  = {} [m]".format(centroid[2]))
        print("<yp> = {} [rad]".format(centroid[3]))
        print("<z>  = {} [m]".format(centroid[4]))
        print("<dE> = {} [GeV]".format(centroid[5]))
    return bunch


def set_centroid(bunch, centroid=0.0, verbose=False):
    if centroid is None:
        return bunch
    if np.ndim(centroid) == 0:
        centroid = 6 * [centroid]
    if all([x is None for x in centroid]):
        return bunch
    old_centroid = get_centroid(bunch)
    for i in range(len(centroid)):
        if centroid[i] is None:
            centroid[i] = old_centroid[i]
    delta = np.subtract(centroid, old_centroid)
    return shift_centroid(bunch, delta=delta, verbose=verbose)


def load(
    filename=None,
    file_format="pyorbit",
    bunch=None,
    verbose=True,
):
    """
    Parameters
    ----------
    filename : str
        Path the file.
    file_format : str
        "pyorbit":
            The expected header format is:
        "parmteq":
            The expected header format is:
                Number of particles    =
                Beam current           =
                RF Frequency           =
                The input file particle coordinates were written in double precision.
                x(cm)             xpr(=dx/ds)       y(cm)             ypr(=dy/ds)       phi(radian)        W(MeV)
    bunch : Bunch
        Create a new bunch if None, otherwise modify this bunch.

    Returns
    -------
    Bunch
    """
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
    _mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

    if verbose:
        print("(rank {}) loading bunch from file {}".format(_mpi_rank, filename))        
    if not os.path.isfile(filename):
        raise ValueError("File '{}' does not exist.".format(filename))
    if bunch is None:
        bunch = Bunch()
    if file_format == "pyorbit":
        bunch.readBunch(filename)
        return bunch
    
    elif file_format == "parmteq":
        # Read data.
        header = np.genfromtxt(filename, max_rows=3, usecols=[0, 1, 2, 3, 4], dtype=str)
        n = int(header[0, 4])
        current = np.float(header[1, 3])
        freq = np.float(header[2, 3]) * 1e6  # MHz to Hz
        data = np.loadtxt(filename, skiprows=5)

        # Trim off-energy particles.
        kin_energy = np.mean(data[:, 5])  # center energy [MeV]
        idx, = np.where(np.abs(data[:, 5] - kin_energy) < (0.05 * kin_energy))
        bunch.getSyncParticle().kinEnergy(kin_energy * 1.0e-3)

        # Unit conversion.
        x = data[idx, 0] * 1.0e-02  # [cm] -> [m]
        xp = data[idx, 1]
        y = data[idx, 2] * 1.0e-02  # [cm] -> [m]
        yp = data[idx, 3]
        phi = data[idx, 4]  # radians
        z = np.rad2deg(phi) / get_z_to_phase_coeff(bunch, freq=freq)
        dE = (data[idx, 5] - kin_energy) * 1.0e-3  # [MeV] -> [GeV]
        
        # Add particles.
        for i in range(n):
            bunch.addParticle(x[i], xp[i], y[i], yp[i], z[i], dE[i])
    else:
        raise KeyError("Unrecognized format '{}'.".format(file_format))
    if verbose:
        print(
            "(rank {}) bunch loaded (nparts={}, macrosize={})".format(
                _mpi_rank, 
                bunch.getSize(), 
                bunch.macroSize()
            )
        )
    return bunch


def generate_xyz(dist=None, n=0, bunch=None, verbose=True):
    """Generate bunch from an x-xp-y-yp-z-dE distribution generator.
    
    Parameters
    ----------
    dist : object
        Must have method `getCoordinates()` that returns (x, xp, y, yp, z, dE).
    n : int
        The number of particles to generate.
    bunch : Bunch
        If provided, it is repopulated with `dist_gen`.
    verbose : bool
        Whether to use progess bar when filling bunch.
        
    Returns
    -------
    Bunch
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
        
    if verbose:
        print("(rank {}) generating particles".format(_mpi_rank))
    for i in (tqdm(range(n)) if verbose else range(n)):
        (x, xp, y, yp, z, dE) = dist.getCoordinates()
        (x, xp, y, yp, z, dE) = orbit_mpi.MPI_Bcast((x, xp, y, yp, z, dE), data_type, main_rank, _mpi_comm)
        if i % _mpi_size == _mpi_rank:
            bunch.addParticle(x, xp, y, yp, z, dE)
    return bunch 


def generate_xy_z(dist_xy=None, dist_z=None, n=0, bunch=None, verbose=True):
    """Generate bunch from x-xp-y-yp, z-dE distribution generators.
        
    Parameters
    ----------
    dist_xy, dist_z : object
        Must have method `getCoordinates()` that returns (x, xp, y, yp), (z, dE) respectively.
    n : int
        The number of particles to generate.
    bunch : Bunch
        If provided, it is repopulated with `dist_gen`.
    verbose : bool
        Whether to use progess bar when filling bunch.
    ring_length : float
        Length of ring to fill uniformly in z.
        
    Returns
    -------
    Bunch
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
        
    if verbose:
        print("(rank {}) generating particles".format(_mpi_rank))
    for i in (tqdm(range(n)) if verbose else range(n)):
        (x, xp, y, yp) = dist_xy.getCoordinates()
        (x, xp, y, yp) = orbit_mpi.MPI_Bcast((x, xp, y, yp), data_type, main_rank, _mpi_comm)
        
        (z, dE) = dist_z.getCoordinates()
        (z, dE) = orbit_mpi.MPI_Bcast((z, dE), data_type, main_rank, _mpi_comm)
        
        if i % _mpi_size == _mpi_rank:
            bunch.addParticle(x, xp, y, yp, z, dE)
    return bunch 


def generate_x_y_z(dist_x=None, dist_y=None, dist_z=None, n=0, bunch=None, verbose=True):
    """Generate bunch from x-xp, y-yp, z-dE distribution generators.

    Parameters
    ----------
    dist_x, dist_y, dist_z : object
        Must have method `getCoordinates()` that returns (x, xp), (y, yp), (z, dE), respectively.
    n : int
        The number of particles to generate.
    bunch : Bunch
        If provided, it is repopulated with `dist_gen`.
    verbose : bool
        Whether to use progess bar when filling bunch.
    ring_length : float
        Length of ring to fill uniformly in z.
        
    Returns
    -------
    Bunch
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

    if verbose:
        print("(rank {}) generating particles".format(_mpi_rank))
    for i in (tqdm(range(n)) if verbose else range(n)):
        (x, xp) = dist_x.getCoordinates()
        (y, yp) = dist_y.getCoordinates()
        (z, dE) = dist_z.getCoordinates()
        (x, xp) = orbit_mpi.MPI_Bcast((x, xp), data_type, main_rank, _mpi_comm)
        (y, yp) = orbit_mpi.MPI_Bcast((y, yp), data_type, main_rank, _mpi_comm)
        (z, dE) = orbit_mpi.MPI_Bcast((z, dE), data_type, main_rank, _mpi_comm)
        if i % _mpi_size == _mpi_rank:
            bunch.addParticle(x, xp, y, yp, z, dE)
    return bunch 


def generate_rms_equiv_xyz(bunch=None, dist_constructor=None, n=None, verbose=True):
    """Generate rms-equivalent bunch from x-xp-y-yp-z-dE distribution generator.
    
    Parameters
    ----------
    bunch : Bunch
        The bunch object to repopulate.
    dist_constructor : class
        The distribution generator constructor: `dist = dist_constructor(twiss_x, twiss_y, twiss_z)`.
    n : int
        The number of particles to generate. If None, use the global number of particles in `bunch`. 
    verbose : bool
        Whether to use progess bar when filling bunch.
        
    Returns
    -------
    Bunch
    """
    if n is None:
        n = bunch.getSizeGlobal()
    if _mpi_rank == 0:
        print("Forming rms-equivalent bunch from 2D Twiss parameters and {} generator.".format(dist_constructor))
        
    stats = pyorbit_sim.diagnostics.BunchStats(bunch)
    (alpha_x, beta_x, eps_x, alpha_y, beta_y, eps_y, alpha_z, beta_z, eps_z) = stats.effective_twiss()
    bunch = generate_xyz(
        dist=dist_constructor(
            twissX=TwissContainer(alpha_x, beta_x, eps_x),
            twissY=TwissContainer(alpha_y, beta_y, eps_y),
            twissZ=TwissContainer(alpha_z, beta_z, eps_z),
        ),
        n=n, 
        bunch=bunch, 
        verbose=verbose,
    )
    return bunch


def generate_rms_equiv_xy_z(
    bunch=None,
    dist_constructor_xy=None,
    dist_constructor_z=None,
    n=None, 
    verbose=True
):
    """Generate rms-equivalent bunch from x-xp-y-yp-z-dE distribution generator.
    
    Parameters
    ----------
    bunch : Bunch
        The bunch object to repopulate.
    dist_constructor_xy : class
        The transverse distribution generator constructor: `dist = dist_constructor(twiss_x, twiss_y)`.
    dist_constructor_z : class
        The longitudinal distribution generator constructor: `dist = dist_constructor(twiss_z)`.
    n : int
        The number of particles to generate. If None, use the global number of particles in `bunch`. 
    verbose : bool
        Whether to use progess bar when filling bunch.
        
    Returns
    -------
    Bunch
    """
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
    _mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

    if n is None:
        n = bunch.getSizeGlobal()
    if _mpi_rank == 0:
        print("Forming rms-equivalent bunch.")
        print("Transverse generator: {}".format(dist_constructor_xy))
        print("Longitudinal generator: {}".format(dist_constructor_xy))
        
    stats = pyorbit_sim.diagnostics.BunchStats(bunch)
    (alpha_x, beta_x, eps_x, alpha_y, beta_y, eps_y, alpha_z, beta_z, eps_z) = stats.effective_twiss()
    bunch = generate_xy_z(
        dist_xy=dist_constructor_xy(
            twissX=TwissContainer(alpha_x, beta_x, eps_x),
            twissY=TwissContainer(alpha_y, beta_y, eps_y),
        ),
        dist_z=dist_constructor_z(
            TwissContainer(alpha_z, beta_z, eps_z)
        ),
        n=n, 
        bunch=bunch, 
        verbose=verbose,
    )
    return bunch


def generate_norm_twiss(
    dist_constructor=None, 
    n=None, 
    bunch=None, 
    alpha_x=-1.9620,
    alpha_y=1.7681,
    alpha_z=-0.0196,
    beta_x=0.1831,
    beta_y=0.1620,
    beta_z=0.5844,
    eps_x=0.21e-06,
    eps_y=0.21e-06,
    eps_z=0.24153e-06,
    verbose=True,
):
    """Generate 6D bunch from distribution generator and normalized Twiss parameters.
    
    Parameters
    ----------
    dist_constructor : class
        The distribution generator constructor: `dist = dist_constructor(twissX, twissY, twissZ)`.
    n : int
        The number of particles to generate.
    bunch : Bunch
        If provided, it is repopulated with `dist_gen`.
    verbose : bool
        Whether to use progess bar when filling bunch.
    alpha, beta, eps : float
        Normalized Twiss parameters. Defaults are the design parameters at the SNS RFQ exit.
    mass, kin_energy: float
        Mass [GeV / c^2] and kinetic energy [GeV] used to unnormalize the Twiss parameters.
        
    Returns
    -------
    Bunch
    """
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
    _mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)
    
    if verbose and _mpi_rank == 0:
        print("Generating bunch from design Twiss parameters and {} generator.".format(dist))  
        
    if n is None:
        n = int(1.00e+05)
        
    mass = bunch.mass()
    kin_energy = bunch.getSyncParticle().kinEnergy()
    gamma = (mass + kin_energy) / mass
    beta = math.sqrt(gamma * gamma - 1.0) / gamma
    eps_x = eps_x / (beta * gamma)  # [m * rad]
    eps_y = eps_y / (beta * gamma)  # [m * rad]
    eps_z = eps_z / (beta * gamma**3)  # [m * rad]    
    eps_z = eps_z * (gamma**3 * beta**2 * mass)  # [m * GeV]
    beta_z = beta_z / (gamma**3 * beta**2 * mass)    
    bunch = generate_xyz(
        dist=dist_constructor(
            twissX=TwissContainer(alpha_x, beta_x, eps_x),
            twissY=TwissContainer(alpha_y, beta_y, eps_y),
            twissZ=TwissContainer(alpha_z, beta_z, eps_z),
        ),
        bunch=bunch,
        n=n,
        verbose=verbose,
    )    
    return bunch    