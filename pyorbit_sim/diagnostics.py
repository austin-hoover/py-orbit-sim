from __future__ import print_function

import numpy as np

from bunch import Bunch
from bunch import BunchTwissAnalysis
from orbit.bunch_generators import TwissAnalysis
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
            self.twiss_calc.getEffectiveAlpha(0),
            self.twiss_calc.getEffectiveBeta(0),
            self.twiss_calc.getEffectiveEmittance(0),
            self.twiss_calc.getEffectiveAlpha(1),
            self.twiss_calc.getEffectiveBeta(1),
            self.twiss_calc.getEffectiveEmittance(1),
        )
