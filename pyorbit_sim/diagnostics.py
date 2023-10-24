from __future__ import print_function
import os
import sys

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

        if self.limits is None:
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
            self.limits = [limits[k] for k in self.axis]
            
        self.edges = [np.linspace(self.limits[i][0], self.limits[i][1], self.n_bins[i] + 1) for i in range(2)]
        self.coords = [0.5 * (e[:-1] + e[1:]) for e in self.edges]
                
        self.grid = Grid2D(
            self.n_bins[0],
            self.n_bins[1], 
            self.limits[0][0], 
            self.limits[1][1], 
            self.limits[0][0], 
            self.limits[1][1],
        )
        self.grid.binBunch(bunch, self.axis[0], self.axis[1])
        self.grid.synchronizeMPI(_mpi_comm)
        
    def get_hist(self):
        self.hist = np.zeros((self.n_bins[0], self.n_bins[1]))
        for i in range(self.hist.shape[0]):
            for j in range(self.hist.shape[1]):
                self.hist[i, j] = self.grid.getValueOnGrid(i, j)        
        return self.hist, self.coords
    
    
class Histogrammer:
    """Only accepts 2D projections right now."""
    def __init__(
        self,
        axes=None, 
        limits=None, 
        n_bins=None, 
        transform=None,         
        outdir=".", 
        index=0,
        position=0.0,
    ):        
        self.outdir = outdir
        self.index = index
        self.position = position

        self.axes = axes
        
        self.limits = limits
        if self.limits is None:
            self.limits = 6 * [None]
            
        self.n_bins = n_bins
        if self.n_bins is None:
            self.n_bins = 6 * [50]
            
        self.transform = transform
        self.dims = ["x", "xp", "y", "yp", "z", "zp"]

        self.histograms = []
        for axis in self.axes:
            histogram = Histogram2D(
                axis=axis, 
                n_bins=[self.n_bins[k] for k in axis],
                limits=[self.limits[k] for k in axis],
            )
            self.histograms.append(histogram)
        
    def action(
        self, 
        params_dict, 
        index=None,
        node=None,
        position=None,    
        verbose=False,
    ):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
                                
        bunch = params_dict["bunch"]
        new_bunch = Bunch()
        bunch.copyBunchTo(new_bunch)
                
        if self.transform is not None:
            X = pyorbit_sim.bunch_utils.get_coords(new_bunch)
            X = self.transform(X)
            for i in range(X.shape[0]):
                x, xp, y, yp, z, dE = X[i, :]
                new_bunch.x(i, x)
                new_bunch.y(i, y)
                new_bunch.z(i, z)
                new_bunch.xp(i, xp)
                new_bunch.yp(i, yp)
                new_bunch.dE(i, dE)
                
        for histogram in self.histograms:
            histogram.action(new_bunch)
            hist, coords = histogram.get_hist()
            
            data = np.zeros((hist.shape[0] + 1, hist.shape[1] + 1))
            data[1:, 0] = coords[0]
            data[0, 1:] = coords[1]
            data[1:, 1:] = hist
            
            filename = "hist_{}-{}".format(histogram.axis[0], histogram.axis[1])
            filename = "{}_{:05.0f}".format(filename, index if index is not None else self.index)
            if node is not None:
                filename = "{}_{}".format(filename, node)
            filename = "{}.npy".format(filename)
            filename = os.path.join(self.outdir, filename)
            if _mpi_rank == 0:
                print("Saving file {}".format(filename))
            np.save(filename, data)
            
        if position is not None:
            self.position = position
