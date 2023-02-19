from __future__ import print_function

import numpy as np
import pandas as pd
import scipy.optimize

from bunch import Bunch
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit_utils import Matrix


def get_matrix_lattice(lattice, mass=None, kin_energy=None):
    test_bunch = Bunch()
    test_bunch.mass(mass)
    test_bunch.getSyncParticle().kinEnergy(kin_energy)
    return TEAPOT_MATRIX_Lattice(lattice, test_bunch)


def get_matrix_lattice_twiss(matrix_lattice):    
    (pos_nu_x, pos_alpha_x, pos_beta_x) = matrix_lattice.getRingTwissDataX()
    (pos_nu_y, pos_alpha_y, pos_beta_y) = matrix_lattice.getRingTwissDataY()
    data = dict()
    data['s'] = np.array(pos_nu_x)[:, 0]
    data['nu_x'] = np.array(pos_nu_x)[:, 1]
    data['nu_y'] = np.array(pos_nu_y)[:, 1]
    data['alpha_x'] = np.array(pos_alpha_x)[:, 1]
    data['alpha_y'] = np.array(pos_alpha_y)[:, 1]
    data['beta_x'] = np.array(pos_beta_x)[:, 1]
    data['beta_y'] = np.array(pos_beta_y)[:, 1]
    keys = ['s', 'nu_x', 'alpha_x', 'beta_x', 'nu_y', 'alpha_y', 'beta_y']
    data = np.vstack([data[key] for key in keys]).T
    return pd.DataFrame(data, columns=keys)


def get_matrix_lattice_dispersion(matrix_lattice):
    (pos_disp_x, pos_disp_p_x) = matrix_lattice.getRingDispersionDataX()
    (pos_disp_y, pos_disp_p_y) = matrix_lattice.getRingDispersionDataY()
    data = dict()
    data['s'] = np.array(pos_disp_x)[:, 0]
    data['disp_x'] = np.array(pos_disp_x)[:, 1]
    data['disp_y'] = np.array(pos_disp_y)[:, 1]
    data['disp_p_x'] = np.array(pos_disp_p_x)[:, 1]
    data['disp_p_y'] = np.array(pos_disp_p_y)[:, 1]
    keys = ['s', 'disp_x', 'disp_p_x', 'disp_y', 'disp_p_y']
    data = np.vstack([data[key] for key in keys]).T
    return pd.DataFrame(data, columns=keys)


def get_sublattice(lattice, start=None, stop=None):   
    def get_index(arg, default=0):
        if arg is None:
            return default
        if type(arg) is str:
            return lattice.getNodeIndex(lattice.getNodeForName(arg))
        else:
            return arg
    start_index = get_index(start, default=0)
    stop_index = get_index(stop, default=-1)
    return lattice.getSubLattice(start_index, stop_index)


def split_node(node, max_part_length=None):
    if max_part_length is not None:
        if node.getLength() > max_part_length:
            node.setnParts(1 + int(node.getLength() / max_part_length))
            

def set_node_fringe(node=None, setting=False, verbose=False):
    try:
        node.setUsageFringeFieldIN(setting)
        node.setUsageFringeFieldOUT(setting)
        if verbose:
            print('Set {} fringe {}'.format(node.getName(), setting))
    except:
        if verbose:
            print('{} does not have setUsageFringeField method.'.format(node.getName()))
            
            
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
    result = scipy.optimize.least_squares(cost, k0, args=(correct_phase_adv,))
    k1, k2 = result.x
    return fodo(k1, k2)