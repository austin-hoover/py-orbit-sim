from __future__ import print_function

import numpy as np

import orbit_mpi
from bunch import Bunch
from bunch import BunchTwissAnalysis
from orbit.bunch_generators import TwissAnalysis
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.teapot import teapot
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.utils import consts
from orbit_mpi import mpi_comm
from orbit_mpi import mpi_datatype
from orbit_mpi import mpi_op
from orbit_utils import BunchExtremaCalculator
from orbit_utils import Matrix
from spacecharge import Grid2D

import pyorbit_sim


class BunchStats:
    def __init__(self, bunch=None, dispersion_flag=0, emit_norm_flag=0):
        self.order = 2
        self.twiss_calc = BunchTwissAnalysis()
        self.twiss_calc.computeBunchMoments(
            bunch, 
            self.order, 
            int(dispersion_flag), 
            int(emit_norm_flag)
        )
        
    def mean(self):
        return np.array([self.twiss_calc.getAverage(i) for i in range(6)])
    
    def cov(self):
        cov = np.zeros((6, 6))
        for i in range(6):
            for j in range(i + 1):
                value = self.twiss_calc.getCorrelation(j, i)
                cov[i, j] = cov[j, i] = value
        return cov
    
    def effective_twiss(self):    
        eps_x = self.twiss_calc.getEffectiveEmittance(0)
        eps_y = self.twiss_calc.getEffectiveEmittance(1)
        eps_z = self.twiss_calc.getEffectiveEmittance(2)
        beta_x = self.twiss_calc.getEffectiveBeta(0)
        beta_y = self.twiss_calc.getEffectiveBeta(1)
        beta_z = self.twiss_calc.getEffectiveBeta(2)
        alpha_x = self.twiss_calc.getEffectiveAlpha(0)
        alpha_y = self.twiss_calc.getEffectiveAlpha(1)
        alpha_z = self.twiss_calc.getEffectiveAlpha(2) 
        return (
            alpha_x, beta_x, eps_x,
            alpha_y, beta_y, eps_y, 
            alpha_z, beta_z, eps_z,
        )

    def effective_twiss_xy(self):    
        return (
            self.twiss_calc.getEffectiveAlpha(0),
            self.twiss_calc.getEffectiveBeta(0),
            self.twiss_calc.getEffectiveEmittance(0),
            self.twiss_calc.getEffectiveAlpha(1),
            self.twiss_calc.getEffectiveBeta(1),
            self.twiss_calc.getEffectiveEmittance(1),
        )

    
class Histogram2D:
    def __init__(self, axis=(0, 1), n_bins=(50, 50), limits=None):
        self.axis = axis
        self.n_bins = n_bins
        self.limits = limits
        self.grid = None
        self.hist = None
        self.edges = None
        self.coords = None
        
    def action(self, bunch):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
        _mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

        limits = self.limits
        if limits is None:
            calc = BunchExtremaCalculator()
            (x_min, x_max, y_min, y_max, z_min, z_max) = calc.extremaXYZ(bunch)
            (xp_min, xp_max, yp_min, yp_max, dE_min, dE_max) = calc.extremaXpYpdE(bunch)
            limits = [
                (x_min, x_max),
                (xp_min, xp_max),
                (y_min, y_max),
                (yp_min, yp_max),
                (z_min, z_max),
                (dE_min, dE_max),
            ]
            
        self.limits = limits
        self.edges = [np.linspace(limits[i][0], limits[i][1], self.n_bins[i] + 1) for i in range(2)]
        self.coords = [0.5 * (e[:-1] + e[1:]) for e in self.edges]
        
        self.grid = Grid2D(
            self.n_bins[0],
            self.n_bins[1], 
            limits[self.axis[0]][0], 
            limits[self.axis[0]][1], 
            limits[self.axis[1]][0], 
            limits[self.axis[1]][1],
        )
        self.grid.binBunch(bunch, self.axis[0], self.axis[1])
        self.grid.synchronizeMPI(_mpi_comm)
        
    def get_hist(self):
        self.hist = np.zeros((self.n_bins[0], self.n_bins[1]))
        for i in range(self.hist.shape[0]):
            for j in range(self.hist.shape[1]):
                self.hist[i, j] = self.grid.getValueOnGrid(i, j)        
        return self.hist, self.edges