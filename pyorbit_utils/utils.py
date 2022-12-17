from orbit.consts import consts


def orbit_matrix_to_numpy(matrix):
    """Return ndarray from two-dimensional orbit matrix."""
    array = np.zeros(matrix.size())
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i, j] = matrix.get(i, j)
    return array


def get_Brho(mass, kin_energy):
    """Compute magnetic rigidity (B rho).

    Parameters
    ----------
    mass : float
        Particle mass [GeV/c^2].
    kin_energy : float
        Particle kinetic energy [GeV].

    Returns
    -------
    float
        Magnetic rigidity [T * m].
    """
    pc = get_pc(mass, kin_energy)
    return 1e9 * (pc / speed_of_light)


def get_pc(mass, kin_energy):
    """Compute momentum * speed_of_light [GeV].

    Parameters
    ----------
    mass : float
        Particle mass [GeV/c^2].
    kin_energy : float
        Particle kinetic energy [GeV].
    """
    return np.sqrt(kin_energy * (kin_energy + 2 * mass))


def get_perveance(mass=None, kin_energy=None, line_density=None):
    """Compute dimensionless space charge perveance from intensity.

    Parameters
    ----------
    mass : float
        Mass per particle [GeV/c^2].
    kin_energy : float
        Kinetic energy per particle [GeV].
    line_density : float
        Number density in longitudinal direction [m^-1].

    Returns
    -------
    float
        Dimensionless space charge perveance.
    """
    gamma = 1 + (kin_energy / mass)  # Lorentz factor
    beta = np.sqrt(1 - (1 / gamma) ** 2)  # velocity/speed_of_light
    return (2 * consts.classical_proton_radius * line_density) / (beta**2 * gamma**3)


def get_intensity(perveance=None, mass=None, kin_energy=None, bunch_length=None):
    """Compute intensity from space charge perveance.

    Parameters
    ----------
    perveance : float
        Dimensionless space charge perveance.
    mass : float
        Mass per particle [GeV/c^2].
    kin_energy : float
        Kinetic energy per particle [GeV].
    bunch_length : float
        The length of the bunch.

    Returns
    -------
    float
        The beam intensity.
    """
    gamma = 1 + (kin_energy / mass)  # Lorentz factor
    beta = np.sqrt(1 - (1 / gamma) ** 2)  # velocity/speed_of_light
    return (
        beta**2
        * gamma**3
        * perveance
        * bunch_length
        / (2 * classical_proton_radius)
    )