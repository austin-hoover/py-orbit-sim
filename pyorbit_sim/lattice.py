from __future__ import print_function

import numpy as np
import pandas as pd

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
        node.setUsageFringeFieldIN(False)
        node.setUsageFringeFieldOUT(False)
        if verbose:
            print('Set {} fringe {}'.format(node.getName(), setting))
    except:
        if verbose:
            print('{} does not have setUsageFringeField method.'.format(node.getName()))