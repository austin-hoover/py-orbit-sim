def add_node_at_lattice_entrance(lattice, new_node):
    """Add node as child at entrance of first node in lattice."""
    firstnode = lattice.getNodes()[0]
    firstnode.addChildNode(new_node, firstnode.ENTRANCE)


def add_node_at_lattice_exit(lattice, new_node):
    """Add node as child at end of last node in lattice."""
    lastnode = lattice.getNodes()[-1]
    lastnode.addChildNode(node, lastnode.EXIT)


def add_node_throughout_lattice(lattice, new_node, location=AccNode.ENTRANCE):
    """Add `new_node` as child of every node in lattice.

    Parameters
    ----------
    lattice : teapot.TEAPOT_Lattice
        Lattice in which node will be inserted.
    new_node : NodeTEAPOT
        Node to insert.
    position : {AccNode.ENTRANCE, AccNode.BODY, AccNode.EXIT}
        Insertation location in each lattice node.

    Returns
    -------
    lattice
    """
    for node in lattice.getNodes():
        node.addChildNode(new_node, location, 0, AccNode.BEFORE)
    return lattice


def get_transfer_matrix(lattice, mass=None, energy=None):
    """Shortcut to get 6 x 6 transfer matrix from periodic lattice.

    Parameters
    ----------
    lattice : teapot.TEAPOT_Lattice
        A periodic lattice to track with.
    mass, energy : float
        Particle mass [GeV/c^2] and kinetic energy [GeV].

    Returns
    -------
    M : ndarray, shape (4, 4)
        Transverse transfer matrix.
    """
    bunch, params_dict = initialize_bunch(mass, energy)
    matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, bunch)
    return utils.orbit_matrix_to_numpy(matrix_lattice.oneTurnMatrix)


def get_matrix_lattice(lattice, mass=None, kin_energy=None):
    single_particle_bunch = initialize_bunch(mass=mass, kin_energy=kin_energy)
    return TEAPOT_MATRIX_Lattice(lattice, single_particle_bunch)


def get_twiss_at_entrance(lattice, mass=None, kin_energy=None):
    """Get 2D Twiss parameters at lattice entrance.

    Parameters
    ----------
    lattice : teapot.TEAPOT_Lattice
        A periodic lattice to track with.
    mass, energy : float
        Particle mass [GeV/c^2] and kinetic energy [GeV].

    Returns
    -------
    alpha_x, alpha_y, beta_x, beta_y : float
        2D Twiss parameters at lattice entrance.
    """
    matrix_lattice = get_matrix_lattice(lattice, mass=mass, kin_energy=kin_energy)
    _, arrPosAlphaX, arrPosBetaX = matrix_lattice.getRingTwissDataX()
    _, arrPosAlphaY, arrPosBetaY = matrix_lattice.getRingTwissDataY()
    alpha_x, alpha_y = arrPosAlphaX[0][1], arrPosAlphaY[0][1]
    beta_x, beta_y = arrPosBetaX[0][1], arrPosBetaY[0][1]
    return alpha_x, alpha_y, beta_x, beta_y


def get_twiss_throughout(lattice, mass=None, kin_energy=None):
    """Get Twiss parameters throughout lattice.

    Parameters
    ----------
    lattice : teapot.TEAPOT_Lattice
        A periodic lattice to track with.
    bunch : Bunch
        Test bunch to perform tracking.

    Returns
    -------
    ndarray
        Columns: [position, nux, nuy, alpha_x, beta_x, alpha_y, beta_y].
    """
    # Track Twiss parameters through linear matrix lattice.
    matrix_lattice = get_matrix_lattice(lattice, mass=mass, kin_energy=kin_energy)
    twiss_x = matrix_lattice.getRingTwissDataX()
    twiss_y = matrix_lattice.getRingTwissDataY()
    # Unpack and convert to ndarrays
    (nux, alpha_x, beta_x), (nuy, alpha_y, beta_y) = twiss_x, twiss_y
    nux, alpha_x, beta_x = np.array(nux), np.array(alpha_x), np.array(beta_x)
    nuy, alpha_y, beta_y = np.array(nuy), np.array(alpha_y), np.array(beta_y)
    # Merge into one array
    s = nux[:, 0]
    nux, alpha_x, beta_x = nux[:, 1], alpha_x[:, 1], beta_x[:, 1]
    nuy, alpha_y, beta_y = nuy[:, 1], alpha_y[:, 1], beta_y[:, 1]
    return np.vstack([s, nux, nuy, alpha_x, beta_x, alpha_y, beta_y]).T


def get_sublattice(lattice, start_node_name=None, stop_node_name=None):
    """Return sublattice from `start_node_name` through `stop_node_name`.

    Parameters
    ----------
    lattice : teapot.TEAPOT_Lattice
        The original lattice from which to create the sublattice.
    start_node_name, stop_node_name : str
        Names of the nodes in the original lattice to use as the first and
        last node in the sublattice.

    Returns
    -------
    teapot.TEAPOT_Lattice
        New lattice consisting of the specified region of the original lattice.
        Note that it is not a copy; changes to the nodes in the new lattice
        affect the nodes in the original lattice.
    """
    if start_node_name is None:
        start_index = 0
    else:
        start_node = lattice.getNodeForName(start_node_name)
        start_index = lattice.getNodeIndex(start_node)
    if stop_node_name is None:
        stop_index = -1
    else:
        stop_node = lattice.getNodeForName(stop_node_name)
        stop_index = lattice.getNodeIndex(stop_node)
    return lattice.getSubLattice(start_index, stop_index)


def split_node(node, max_length=None):
    """Split node into parts so no part is longer than max_node_length."""
    if max_length is not None:
        if node.getLength() > max_node_length:
            node.setnParts(1 + int(node.getLength() / max_length))
    return node
