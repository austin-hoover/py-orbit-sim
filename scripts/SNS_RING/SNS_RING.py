"""SNS ring lattice generation."""
from __future__ import print_function
import os
import sys
import time

from bunch import Bunch
from foil import Foil
from impedances import LImpedance
from impedances import TImpedance
from orbit.aperture import CircleApertureNode
from orbit.aperture import EllipseApertureNode
from orbit.aperture import RectangleApertureNode
from orbit.aperture import TeapotApertureNode
from orbit.bumps import bumps
from orbit.bumps import BumpLatticeModifications
from orbit.bumps import TeapotBumpNode
from orbit.bumps import TeapotSimpleBumpNode
from orbit.collimation import addTeapotCollimatorNode
from orbit.collimation import TeapotCollimatorNode
from orbit.envelope import DanilovEnvelope
from orbit.foils import addTeapotFoilNode
from orbit.foils import TeapotFoilNode
from orbit.impedances import addImpedanceNode
from orbit.impedances import BetFreqDep_LImpedance_Node
from orbit.impedances import BetFreqDep_TImpedance_Node
from orbit.impedances import FreqDep_LImpedance_Node
from orbit.impedances import FreqDep_TImpedance_Node
from orbit.impedances import LImpedance_Node
from orbit.impedances import TImpedance_Node
from orbit.injection import addTeapotInjectionNode
from orbit.injection import InjectParts
from orbit.injection import TeapotInjectionNode
from orbit.injection.distributions import JohoLongitudinal
from orbit.injection.distributions import JohoTransverse
from orbit.injection.distributions import SNSESpreadDist
from orbit.injection.distributions import UniformLongDist
from orbit.lattice import AccNode
from orbit.lattice import AccLattice
from orbit.rf_cavities import RFNode
from orbit.rf_cavities import RFLatticeModifications
from orbit.space_charge import sc2p5d
from orbit.space_charge import sc2dslicebyslice
from orbit.space_charge.envelope import set_env_solver_nodes
from orbit.space_charge.sc1d import addLongitudinalSpaceChargeNode
from orbit.space_charge.sc1d import SC1D_AccNode
from orbit.space_charge.sc2dslicebyslice.scLatticeModifications import (
    setSC2DSliceBySliceAccNodes,
)
from orbit.space_charge.sc2p5d.scLatticeModifications import setSC2p5DAccNodes
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice_Coupled
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_Ring
from orbit.time_dep import TIME_DEP_Lattice
from orbit.time_dep.waveform import ConstantWaveform
from orbit.time_dep.waveform import SquareRootWaveform
from orbit.utils import consts
import orbit_mpi
from spacecharge import LSpaceChargeCalc
from spacecharge import Boundary2D
from spacecharge import SpaceChargeCalc2p5D
from spacecharge import SpaceChargeCalcSliceBySlice2D


# Global variables
X_FOIL = 0.0486  # [m]
Y_FOIL = 0.0460  # [m]


class SNS_RING(TIME_DEP_Lattice):
    """CLass for SNS ring simulations.

    Note that TEAPOT_Ring has been changed: children are added on instantiation.
    This means that the nodes cannot be split. This is not a problem for
    TEAPOT_Lattice.
    """

    def __init__(
        self,
        nominal_n_inj_turns=1000,
        nominal_intensity=1.5e14,
        nominal_bunch_length_frac=(50.0 / 64.0),
    ):
        TIME_DEP_Lattice.__init__(self)
        self.nominal_n_inj_turns = nominal_n_inj_turns
        self.nominal_intensity = nominal_intensity
        self.nominal_bunch_length_frac = nominal_bunch_length_frac
        self.bunch = None
        self.sync_parts = None
        self.lostbunch = None
        self.params_dict = None
        self.collimator_node = None
        self.foil_node = None
        self.inj_node = None
        self.rf_nodes = None
        
    def init_bunch(self, mass=0.938, kin_energy=1.0):
        self.bunch = Bunch()
        self.bunch.mass(mass)
        self.sync_part = self.bunch.getSyncParticle()
        self.sync_part.kinEnergy(kin_energy)
        self.lostbunch = Bunch()
        self.lostbunch.addPartAttr('LostParticleAttributes')
        self.params_dict = {'bunch': self.bunch, 'lostbunch': self.lostbunch}

    def add_collimator_node(
        self,
        angle=0.0,
        density_fac=1.0,
        length=0.00001,
        ma=9,
        a=0.110,
        b=0.0,
        c=0.0,
        d=0.0,
        position=0.5,
        shape=1,
        name="collimator1",
    ):
        """Add black absorber collimator to act as an aperture."""
        self.collimator_node = TeapotCollimatorNode(
            length, ma, density_fac, shape, a, b, c, d, angle, position, name
        )
        addTeapotCollimatorNode(self, position, self.collimator_node)

    def add_foil_node(
        self,
        xmin=-0.0085,
        xmin=+0.0085,
        ymin=-0.0080,
        ymax=+0.1000,
        thickness=390.0,
        scatter='full',  # {'full', 'simple'}
        name='foil node',
    ):
        """Add foil scattering node."""
        xmin += X_FOIL
        xmax += X_FOIL
        ymin += Y_FOIL
        ymax += Y_FOIL
        scatter = {'full': 0, 'simple': 1}[scatter]  
        self.foil_node = TeapotFoilNode(xmin, xmax, ymin, ymax, thickness)
        self.foil_node.setScatterChoice(scatter)
        self.getNodes()[0].addChildNode(self.foil_node, AccNode.ENTRANCE)

    def add_inj_node(
        self,     
        n_parts=1, 
        n_parts_max=None,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        dist_x_kws=None,
        dist_y_kws=None,
        dist_z_kws=None,
        **kws
    ):
        """Add injection node."""
        if n_parts_max is None:
            n_parts_max = -1
        if xmin is None:
            xmin = -np.inf
        if xmax is None:
            xmax = +np.inf
        if ymin is None:
            ymin = -np.inf
        if ymax is None:
            ymax = +np.inf
        dist_x_kws_default = {
            'centerpos': X_FOIL,
            'centermom': 0.0,
            'alpha': -0.924,
            'beta': 3.71,
            'order': 9.0,
            'eps_rms': 0.467e-6,
        }
        dist_y_kws_default = {
            'centerpos': Y_FOIL,
            'centermom': 0.0,
            'alpha': -0.5,
            'beta': 4.86,
            'order': 9.0,
            'eps_rms': 0.300e-6,
        }
        dist_z_kws_default = {
            'lattlength': self.getLength(),
            'zmin': -self.nominal_bunch_length_frac * self.getLength(),
            'zmax': +self.nominal_bunch_length_frac * self.getLength(),
            'tailfraction': 0.0,
            'sp': self.sync_part,
            'emean': self.sync_part.kinEnergy(),
            'esigma': 0.0005,
            'etrunc': 1.0,
            'emin': self.sync_part.kinEnergy() - 0.0025,
            'emax': self.sync_part.kinEnergy() + 0.0025,
            'ecparams': {
                'mean': 0.0,
                'sigma': 0.000000001,
                'trunc': 1.0,
                'min': -0.0035,
                'max': +0.0035,
                'drifti': 0.0,
                'driftf': 0.0,
                'drifttime': 1000.0 * self.nominal_n_inj_turns * (self.getLength() / (self.sync_part.beta() * consts.speed_of_light)),
            },
            'esparams': {
                'nu': 100.0,
                'phase': 0.0,
                'max': 0.0,
                'nulltime': 0.0,
            },
        }
        if dist_x_kws is None:
            dist_x_kws = dict()
        if dist_y_kws is None:
            dist_y_kws = dict()
        if dist_z_kws is None:
            dist_z_kws = dict()
        for key, value in dist_x_kws_default.items():
            dist_x_kws.setdefault(key, value)
        for key, value in dist_y_kws_default.items():
            dist_y_kws.setdefault(key, value)
        for key, value in dist_z_kws_default.items():
            dist_z_kws.setdefault(key, value)
            
        dist_x_kws['emitlim'] * dist_x_kws['eps_rms'] * 2.0 * (dist_x_kws['order'] + 1.0)
        dist_y_kws['emitlim'] * dist_y_kws['eps_rms'] * 2.0 * (dist_y_kws['order'] + 1.0)
    
        dist_x = JohoTransverse(**dist_x_kws)
        dist_y = JohoTransverse(**dist_y_kws)
        dist_z = SNSESpreadDist(**dist_z_kws)
        
        self.inj_node = TeapotInjectionNode(
            nparts=n_parts, 
            bunch=self.bunch, 
            lostbunch=self.lostbunch, 
            injectregion=[xmin, xmax, ymin, ymax],
            xDistFunc=dist_x,
            yDistFunc=dist_y,
            lDistFun=dist_z,
            nmaxmacroparticles=n_parts_max,
            **kws
        )
        self..getNodes()[0].addChildNode(self.inj_node, AccNode.ENTRANCE)

    def add_rf_harmonic_nodes(self, **kws):
        """Add RF harmonic cavity nodes.
        
        **kws
            Dictionary with keys {'RF1a', 'RF1b', 'RF1c', 'RF2'}. Each value
            is a dictionary with keys {'phase', 'hnum', 'voltage'}.
        """        
        z_to_phi = 2.0 * np.pi / self.getLength()
        dE_sync = 0.0
        length = 0.0
        kws_default = {
            'RF1a': {
                'phase': 0.0,
                'hnum': 1.0,
                'voltage': +5.03 * 1.0e-6,  # [GV]
            },
            'RF1b': {
                'phase': 0.0,
                'hnum': 1.0,
                'voltage': 0.0,  # [GV]
            },
            'RF1c': {
                'phase': 0.0,
                'hnum': 1.0,
                'voltage': 0.0,  # [GV]
            },
            'RF2': {
                'phase': 0.0,
                'hnum': 2.0,
                'voltage': -5.03 * 1.0e-6,  # [GV]
            },
        }
        for key, value in kws_default.items():
            kws.setdefault(key, value)
        
        self.rf_nodes = {
            'RF1a': RFNode.Harmonic_RFNode(
                z_to_phi,
                dE_sync,
                kws['RF1a']['hnum'], 
                kws['RF1a']['voltage'], 
                kws['RF1a']['phase'], 
                length,
                name='RF1a',
            ),
            'RF1b': RFNode.Harmonic_RFNode(
                z_to_phi,
                dE_sync,
                kws['RF1b']['hnum'], 
                kws['RF1b']['voltage'], 
                kws['RF1b']['phase'], 
                length,
                name='RF1b',
            ),
            'RF1c': RFNode.Harmonic_RFNode(
                z_to_phi,
                dE_sync,
                kws['RF1c']['hnum'], 
                kws['RF1c']['voltage'], 
                kws['RF1c']['phase'], 
                length,
                name='RF1c',
            ),
            'RF2': RFNode.Harmonic_RFNode(
                z_to_phi,
                dE_sync,
                kws['RF2']['hnum'], 
                kws['RF2']['voltage'], 
                kws['RF2']['phase'], 
                length,
                name='RF2',
            ),
        }
        RFLatticeModifications.addRFNode(self, 184.273, self.rf_nodes['RF1a'])
        RFLatticeModifications.addRFNode(self, 186.571, self.rf_nodes['RF1b'])
        RFLatticeModifications.addRFNode(self, 188.868, self.rf_nodes['RF1c'])
        RFLatticeModifications.addRFNode(self, 191.165, self.rf_nodes['RF2'])
        
    def add_longitudinal_impedance_node(
        self, phase_length=None, n_macros_min=1000, n_bins=128, position=124.0,
        ZL_Ekicker=None, ZL_RF=None,
    ):
        """Add longitudinal impedance node at lattice midpoint.
        
        SNS Longitudinal Impedance tables. EKicker impedance from private communication 
        with J.G. Wang. Seems to be for 7 of the 14 kickers (not sure why). Impedance 
        units are [Ohms / n]. Kicker and RF impedances are inductive with real part 
        positive and imaginary part negative (Chao definition).
        
        TODO: Move these to a text file or a global variable.
        """
        if ZL_Ekicker is None:
            ZL_Ekicker = [
                complex(42.0, -182),
                complex(35, -101.5),
                complex(30.3333, -74.6667),
                complex(31.5, -66.5),
                complex(32.2, -57.4),
                complex(31.5, -51.333),
                complex(31, -49),
                complex(31.5, -46.375),
                complex(31.8889, -43.556),
                complex(32.9, -40.6),
                complex(32.7273, -38.18),
                complex(32.25, -35.58),
                complex(34.46, -32.846),
                complex(35, -30.5),
                complex(35.4667, -28.0),
                complex(36.75, -25.81),
                complex(36.647, -23.88),
                complex(36.944, -21.1667),
                complex(36.474, -20.263),
                complex(36.4, -18.55),
                complex(35.333, -17),
                complex(35, -14.95),
                complex(33.478, -13.69),
                complex(32.375, -11.67),
                complex(30.8, -10.08),
                complex(29.615, -8.077),
                complex(28.519, -6.74),
                complex(27.5, -5),
                complex(26.552, -4.103),
                complex(25.433, -3.266),
                complex(24.3871, -2.7),
                complex(23.40625, -2.18),
            ]
        if ZL_RF is None:
            ZL_RF = [
                complex(0.0, 0.0),
                complex(0.750, 0.0),
                complex(0.333, 0.0),
                complex(0.250, 0.0),
                complex(0.200, 0.0),
                complex(0.167, 0.0),
                complex(3.214, 0.0),
                complex(0.188, 0.0),
                complex(0.167, 0.0),
                complex(0.150, 0.0),
                complex(1.000, 0.0),
                complex(0.125, 0.0),
                complex(0.115, 0.0),
                complex(0.143, 0.0),
                complex(0.333, 0.0),
                complex(0.313, 0.0),
                complex(0.294, 0.0),
                complex(0.278, 0.0),
                complex(0.263, 0.0),
                complex(0.250, 0.0),
                complex(0.714, 0.0),
                complex(0.682, 0.0),
                complex(0.652, 0.0),
                complex(0.625, 0.0),
                complex(0.600, 0.0),
                complex(0.577, 0.0),
                complex(0.536, 0.0),
                complex(0.536, 0.0),
                complex(0.517, 0.0),
                complex(0.500, 0.0),
                complex(0.484, 0.0),
                complex(0.469, 0.0),
            ]
        Z = []
        for zk, zrf in zip(ZL_Ekicker, ZL_RF):
            zreal = (zk.real / 1.75) + zrf.real
            zimag = (zk.imag / 1.75) + zrf.imag
            Z.append(complex(zreal, zimag))
        self.longitudinal_impedance_node = LImpedance_Node(
            phase_length,
            n_macros_min,
            n_bins,
        )
        self.longitudinal_impedance_node.assignImpedance(Z)
        addImpedanceNode(self, position, self.longitudinal_impedance_node)
