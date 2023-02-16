def fodo_lattice(
    mux=80.0,
    muy=80.0,
    L=5.0,
    fill_fac=0.5,
    angle=0.0,
    start="drift",
    fringe=False,
    reverse=False,
):
    """Create a FODO lattice.

    Parameters
    ----------
    mux{y}: float
        The x{y} lattice phase advance [deg]. These are the phase advances
        when the lattice is uncoupled (`angle` == 0).
    L : float
        The length of the lattice [m].
    fill_fac : float
        The fraction of the lattice occupied by quadrupoles.
    angle : float
        The skew or tilt angle of the quads [deg]. The focusing
        quad is rotated clockwise by angle, and the defocusing quad is
        rotated counterclockwise by angle.
    fringe : bool
        Whether to include nonlinear fringe fields in the lattice.
    start : str
        If 'drift', the lattice will be O-F-O-O-D-O. If 'quad' the lattice will
        be (F/2)-O-O-D-O-O-(F/2).
    reverse : bool
        If True, reverse the lattice elements. This places the defocusing quad
        first.

    Returns
    -------
    teapot.TEAPOT_Lattice
    """
    angle = np.radians(angle)

    def fodo(k1, k2):
        """Return FODO lattice.

        k1, k2 : float
            Strengths of the first (focusing) and second (defocusing) quadrupoles.
        """
        # Instantiate elements
        lattice = teapot.TEAPOT_Lattice()
        drift1 = teapot.DriftTEAPOT("drift1")
        drift2 = teapot.DriftTEAPOT("drift2")
        drift_half1 = teapot.DriftTEAPOT("drift_half1")
        drift_half2 = teapot.DriftTEAPOT("drift_half2")
        qf = teapot.QuadTEAPOT("qf")
        qd = teapot.QuadTEAPOT("qd")
        qf_half1 = teapot.QuadTEAPOT("qf_half1")
        qf_half2 = teapot.QuadTEAPOT("qf_half2")
        qd_half1 = teapot.QuadTEAPOT("qd_half1")
        qd_half2 = teapot.QuadTEAPOT("qd_half2")
        # Set lengths
        half_nodes = (drift_half1, drift_half2, qf_half1, qf_half2, qd_half1, qd_half2)
        full_nodes = (drift1, drift2, qf, qd)
        for node in half_nodes:
            node.setLength(L * fill_fac / 4)
        for node in full_nodes:
            node.setLength(L * fill_fac / 2)
        # Set quad focusing strengths
        for node in (qf, qf_half1, qf_half2):
            node.addParam("kq", +k1)
        for node in (qd, qd_half1, qd_half2):
            node.addParam("kq", -k2)
        # Create lattice
        if start == "drift":
            lattice.addNode(drift_half1)
            lattice.addNode(qf)
            lattice.addNode(drift2)
            lattice.addNode(qd)
            lattice.addNode(drift_half2)
        elif start == "quad":
            lattice.addNode(qf_half1)
            lattice.addNode(drift1)
            lattice.addNode(qd)
            lattice.addNode(drift2)
            lattice.addNode(qf_half2)
        # Other
        if reverse:
            lattice.reverseOrder()
        lattice.set_fringe(fringe)
        lattice.initialize()
        for node in lattice.getNodes():
            name = node.getName()
            if "qf" in name:
                node.setTiltAngle(+angle)
            elif "qd" in name:
                node.setTiltAngle(-angle)
        return lattice

    def cost(kvals, correct_tunes, mass=0.93827231, energy=1):
        lattice = fodo(*kvals)
        M = utils.transfer_matrix(lattice, mass, energy)
        tmat = twiss.TransferMatrix(M)
        return correct_phase_adv - 360.0 * np.array(tmat.params['eigtunes'])

    correct_phase_adv = np.array([mux, muy])
    k0 = np.array([0.5, 0.5])  # ~ 80 deg phase advance
    result = opt.least_squares(cost, k0, args=(correct_phase_adv,))
    k1, k2 = result.x
    return fodo(k1, k2)