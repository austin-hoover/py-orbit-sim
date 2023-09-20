"""Helpers for ring simulations."""
from __future__ import print_function
import os
import sys
import time

import numpy as np
import pandas as pd

import orbit_mpi
from bunch import Bunch
from bunch import BunchTwissAnalysis
from bunch_utils_functions import copyCoordsToInitCoordsAttr
from orbit.bunch_utils import ParticleIdNumber
from orbit.lattice import AccActionsContainer
from orbit.matrix_lattice.parameterizations import lebedev_bogacz as LB
from orbit.teapot import DriftTEAPOT
from orbit.utils import consts
from orbit_utils import BunchExtremaCalculator

import pyorbit_sim


_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)



class Monitor:
    def __init__(self, filename=None, verbose=True):
        self.filename = filename
        self.verbose = verbose
        self.start_time = None
        self.iteration = 0
        
        if _mpi_rank == 0:
            keys = [
                "n_parts",
                "gamma",
                "beta",
                "energy",
            ]
            for dim in "xyz":
                keys.append("{}_rms".format(dim))
            for dim in "xyz":
                keys.append("{}_min".format(dim))
                keys.append("{}_max".format(dim))    
            for dim in "xy12":
                keys.append("eps_{}".format(dim))
            for i in range(6):
                keys.append("mean_{}".format(i))
            for i in range(6):
                for j in range(i + 1):
                    keys.append("cov_{}-{}".format(j, i))
                    
            self.history = dict()
            for key in keys:
                self.history[key] = None

            self.filename = filename
            self.file = None
            if self.filename is not None:
                self.file = open(self.filename, "w")
                line = ",".join(list(self.history))
                line = line[:-1] + "\n"
                self.file.write(line)
    
    def action(self, params_dict):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
                
        if self.start_time is None:
            self.start_time = time.clock()
        
        # Get the bunch.
        bunch = params_dict["bunch"]
        beta = bunch.getSyncParticle().beta()
        gamma = bunch.getSyncParticle().gamma()
        n_parts = bunch.getSizeGlobal()
        if _mpi_rank == 0:
            self.history["n_parts"] = n_parts
            self.history["gamma"] = gamma
            self.history["beta"] = beta
            self.history["energy"] = bunch.getSyncParticle().kinEnergy()

        # Measure bunch centroid.
        twiss_analysis = BunchTwissAnalysis()
        order = 2
        dispersion_flag = 0
        emit_norm_flag = 0
        twiss_analysis.computeBunchMoments(bunch, order, dispersion_flag, emit_norm_flag)
        for i in range(6):
            key = "mean_{}".format(i)
            value = twiss_analysis.getAverage(i)
            if _mpi_rank == 0:
                self.history[key] = value
                
        # Measure the covariance matrix.
        Sigma = np.zeros((6, 6))
        for i in range(6):
            for j in range(i + 1):
                key = "cov_{}-{}".format(j, i)
                value = twiss_analysis.getCorrelation(j, i)
                if _mpi_rank == 0:
                    self.history[key] = value
                Sigma[j, i] = Sigma[i, j] = value
          
        # Compute the rms emittances.
        (eps_x, eps_y) = pyorbit_sim.stats.apparent_emittances(Sigma[:4, :4])
        (eps_1, eps_2) = pyorbit_sim.stats.intrinsic_emittances(Sigma[:4, :4])   
        if _mpi_rank == 0:
            self.history["eps_x"] = eps_x
            self.history["eps_y"] = eps_y
            self.history["eps_1"] = eps_1
            self.history["eps_2"] = eps_2
        
        # Compute the rms beam size.
        if _mpi_rank == 0:
            x_rms = np.sqrt(self.history["cov_0-0"])
            y_rms = np.sqrt(self.history["cov_2-2"])
            z_rms = np.sqrt(self.history["cov_4-4"])
            self.history["x_rms"] = x_rms
            self.history["y_rms"] = y_rms
            self.history["z_rms"] = z_rms
            
        # Compute maximum phase space amplitudes.
        extrema_calculator = BunchExtremaCalculator()
        (x_min, x_max, y_min, y_max, z_min, z_max) = extrema_calculator.extremaXYZ(bunch)
        if _mpi_rank == 0:
            self.history["x_min"] = x_min
            self.history["x_max"] = x_max
            self.history["y_min"] = y_min
            self.history["y_max"] = y_max
            self.history["z_min"] = z_min
            self.history["z_max"] = z_max
                      
        # Print update message.
        if self.verbose and _mpi_rank == 0:
            message = "turn={:05.0f} t={:0.3f} nparts={:05.0f}".format(
                self.iteration,
                time.clock() - self.start_time,
                n_parts,
            )
            message = "{} xrms={:0.2e} yrms={:0.2e} epsx={:0.2e} epsy={:0.2e} eps1={:0.2e} eps2={:0.2e} ".format(
                message,
                1.00e+03 * x_rms,
                1.00e+03 * y_rms,
                1.00e+06 * eps_x,
                1.00e+06 * eps_y,
                1.00e+06 * eps_1,
                1.00e+06 * eps_2,
            )
            print(message)

        # Add one line to the history file.
        if _mpi_rank == 0 and self.file is not None:
            data = [self.history[key] for key in self.history]
            line = ""
            for i in range(len(data)):
                line += "{},".format(data[i])
            line = line[:-1] + "\n"
            self.file.write(line)

        self.iteration += 1    


def match_bunch(bunch=None, M=None):
    """Match the covariance matrix to the ring using the transfer matrix eigenvectors.
    
    X -> V inv(W) X, where V is the lattice normalization matrix and 
    W is the bunch normalization matrix.
    
    W transforms the bunch such that Sigma = diag(eps_1, eps_1, eps_2, eps_2), where
    eps_j is the intrinsic emittance of mode j.
    
    TO DO: kv, danilov envelope matching options.
    
    Parameters
    ----------
    M : ndarray, shape (4, 4)
        The one-turn transfer matrix.
    """
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
    
    # Compute lattice normalization matrix V.
    M = M[:4, :4]
    eigenvalues, eigenvectors = np.linalg.eig(M)
    eigenvectors = LB.normalize(eigenvectors)
    V = LB.normalization_matrix_from_eigenvectors(eigenvectors)

    # Compute bunch normalization matrix W.
    stats = pyorbit_sim.diagnostics.BunchStats(bunch)
    Sigma = stats.cov()
    Sigma = Sigma[:4, :4]
    U = LB.unit_symplectic_matrix(4)
    SU = np.matmul(Sigma, U)
    eigenvalues, eigenvectors = np.linalg.eig(SU)
    eigenvectors = LB.normalize(eigenvectors)
    W = LB.normalization_matrix_from_eigenvectors(eigenvectors)
    
    matrix = np.identity(6)
    matrix[:4, :4] = np.matmul(V, np.linalg.inv(W))
    bunch = pyorbit_sim.bunch_utils.linear_transform(bunch, matrix)
    return bunch