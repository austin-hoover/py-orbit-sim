from __future__ import print_function

import numpy as np

from bunch import Bunch
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit_utils import Matrix


def get_matrix_lattice(lattice, mass=None, kin_energy=None):
    bunch = Bunch()
    bunch.mass(mass)
    bunch.getSyncParticle().kinEnergy(kin_energy)
    return TEAPOT_MATRIX_Lattice(lattice, test_bunch)


def unpack_matrix_lattice_twiss(matrix_lattice):
    twiss_x = matrix_lattice.getRingTwissDataX()
    twiss_y = matrix_lattice.getRingTwissDataY()
    pos_nu_x, pos_alpha_x, pos_beta_x = twiss_x
    pos_nu_y, pos_alpha_y, pos_beta_y = twiss_y
    pos_nu_x = np.array(pos_nu_x)
    pos_nu_y = np.array(pos_nu_y)
    pos_alpha_x = np.array(pos_alpha_x)
    pos_alpha_y = np.array(pos_alpha_y)
    pos_beta_x = np.array(pos_beta_x)
    pos_beta_y = np.array(pos_beta_y)
    pos = pos_nu_x[:, 0]
    nu_x = pos_nu_x[:, 1]
    nu_y = pos_nu_y[:, 1]
    alpha_x = pos_alpha_x[:, 1]
    alpha_y = pos_alpha_y[:, 1]
    beta_x = pos_beta_x[:, 1]
    beta_y = pos_beta_y[:, 1]
    return np.vstack([pos, nu_x, nu_y, alpha_x, beta_x, alpha_y, beta_y]).T


def get_sublattice(lattice, first_node_name=None, last_node_name=None):
    if first_node_name is None:
        start_index = 0
    else:
        start_index = lattice.getNodeIndex(lattice.getNodeForName(first_node_name))
    if last_node_name is None:
        stop_index = -1
    else:
        stop_index = lattice.getNodeIndex(lattice.getNodeForName(last_node_name))
    return lattice.getSubLattice(start_index, stop_index)


def split_node(node, max_part_length=None):
    if max_length is not None:
        if node.getLength() > max_part_length:
            node.setnParts(1 + int(node.getLength() / max_part_length))
    return node
