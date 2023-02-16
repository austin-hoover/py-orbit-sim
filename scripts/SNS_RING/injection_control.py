"""This module provides control of the SNS injection region."""
from __future__ import print_function
import sys
import time

import numpy as np
from scipy import optimize as opt
import pandas as pd

from bunch import Bunch
from orbit.diagnostics.diagnostics import BunchCoordsNode
from orbit.diagnostics.diagnostics_lattice_modifications import add_diagnostics_node_as_child
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode

sys.path.append("/home/46h/repo/py-orbit-sim/")
from pyorbit_sim.bunch_utils import initialize_bunch
from pyorbit_sim.lattice import get_sublattice
from pyorbit_sim.misc import get_pc


# Maximum injection kicker angles at 1.0 GeV [mrad]
MIN_KICKER_ANGLES = 1.15 * np.array([0.0, 0.0, -7.13, -7.13, -7.13, -7.13, 0.0, 0.0])
MAX_KICKER_ANGLES = 1.15 * np.array([12.84, 12.84, 0.0, 0.0, 0.0, 0.0, 12.84, 12.84])

# Maximum injection-region dipole corrector angle [mrad]
MAX_CORRECTOR_ANGLE = 1.5  # [mrad]


class InjRegionController:
    """Control the closed orbit in the injection region."""

    def __init__(
        self,
        ring=None,
        mass=None,
        kin_energy=None,
        scale=None,
        inj_mid="injm1",
        inj_start="bpm_a09",
        inj_end="bpm_b01",
    ):
        self.ring = ring
        self.mass = mass
        self.kin_energy = kin_energy
        self.kicker_names = [
            "ikickh_a10",
            "ikickv_a10",
            "ikickh_a11",
            "ikickv_a11",
            "ikickv_a12",
            "ikickh_a12",
            "ikickv_a13",
            "ikickh_a13",
        ]
        self.kicker_nodes = [ring.getNodeForName(name) for name in self.kicker_names]

        # Maximum injection kicker angles at 1.0 GeV [mrad].
        self.min_kicker_angles = MIN_KICKER_ANGLES
        self.max_kicker_angles = MAX_KICKER_ANGLES

        # Convert the maximum kicker angles from [mrad] to [rad].
        self.min_kicker_angles = 1.0e-3 * self.min_kicker_angles
        self.max_kicker_angles = 1.0e-3 * self.max_kicker_angles

        # Scale the maximum kicker angles based on kinetic energy. (They are defined for
        # the nominal SNS kinetic energy of 1.0 GeV.)
        self.kin_energy_scale_factor = get_pc(mass, kin_energy=1.0) / get_pc(mass, kin_energy)
        self.min_kicker_angles *= self.kin_energy_scale_factor
        self.max_kicker_angles *= self.kin_energy_scale_factor

        # Artificially scale the maximum kicker angles.
        if scale is not None:
            print("Scaling maximum kicker angles by factor {}".format(scale))
            self.min_kicker_angles = scale * self.min_kicker_angles
            self.max_kicker_angles = scale * self.max_kicker_angles

        # Identify the horizontal and vertical kicker nodes. (PyORBIT does not
        # distinguish between horizontal/vertical kickers.)
        self.kicker_idx_x = [0, 2, 5, 7]
        self.kicker_idx_y = [1, 3, 4, 6]
        self.kicker_nodes_x = [self.kicker_nodes[i] for i in self.kicker_idx_x]
        self.kicker_nodes_y = [self.kicker_nodes[i] for i in self.kicker_idx_y]
        self.min_kicker_angles_x = self.min_kicker_angles[self.kicker_idx_x]
        self.max_kicker_angles_x = self.max_kicker_angles[self.kicker_idx_x]
        self.min_kicker_angles_y = self.min_kicker_angles[self.kicker_idx_y]
        self.max_kicker_angles_y = self.max_kicker_angles[self.kicker_idx_y]

        # Identify dipole correctors. These will be used to make a closed bump.
        self.corrector_names = ["dmcv_a09", "dchv_a10", "dchv_a13", "dmcv_b01"]
        self.corrector_nodes = [self.ring.getNodeForName(name) for name in self.corrector_names]
        self.max_corrector_angle = MAX_CORRECTOR_ANGLE  # [mrad]
        self.max_corrector_angle *= 1.0e-3  # [mrad] --> [rad]
        self.max_corrector_angle *= self.kin_energy_scale_factor
        self.min_corrector_angle = -self.max_corrector_angle
        self.min_corrector_angles_y = np.full(4, self.min_corrector_angle)
        self.max_corrector_angles_y = np.full(4, self.max_corrector_angle)

        # Create one sublattice for the first half of the injection region (before the foil)
        # and one sublattice for the second half of the injection region (afterthe foil).
        self.sublattice1 = get_sublattice(self.ring, inj_start, None)
        self.sublattice2 = get_sublattice(self.ring, inj_mid, inj_end)
        self.sublattices = [self.sublattice1, self.sublattice2]
        
        # Add inactive monitor nodes
        self.monitor_nodes = []
        for sublattice in self.sublattices:
            self.monitor_nodes.append([])
            node_pos_dict = sublattice.getNodePositionsDict()
            for node in sublattice.getNodes():
                monitor_node = BunchCoordsNode(axis=(0, 1, 2, 3))
                monitor_node.position = node_pos_dict[node][0]
                monitor_node.active = False
                add_diagnostics_node_as_child(
                    monitor_node,
                    parent_node=node,
                    part_index=0,
                    place_in_part=AccActionsContainer.BEFORE,
                )
                self.monitor_nodes[-1].append(monitor_node)
                
        # Initialize bunch for single-particle tracking.
        self.bunch = Bunch()
        self.bunch.mass(self.mass)
        self.bunch.getSyncParticle().kinEnergy(self.kin_energy)
        self.params_dict = {'lostbunch': Bunch()}  # for apertures
        self.bunch.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def get_kicker_angles_x(self):
        return np.array([node.getParam("kx") for node in self.kicker_nodes_x])

    def get_kicker_angles_y(self):
        return np.array([node.getParam("ky") for node in self.kicker_nodes_y])

    def get_kicker_angles(self):
        angles = []
        for node in self.kicker_nodes:
            if node in self.kicker_nodes_x:
                angles.append(node.getParam("kx"))
            elif node in self.kicker_nodes_y:
                angles.append(node.getParam("ky"))
        return np.array(angles)

    def set_kicker_angles_x(self, angles=None):
        if angles is not None:
            for angle, node in zip(angles, self.kicker_nodes_x):
                node.setParam("kx", angle)

    def set_kicker_angles_y(self, angles=None):
        if angles is not None:
            for angle, node in zip(angles, self.kicker_nodes_y):
                node.setParam("ky", angle)
                
    def set_kicker_angles(self, angles):
        angles_x = [angles[i] for i in self.kicker_idx_x]
        angles_y = [angles[i] for i in self.kicker_idx_y]
        self.set_kicker_angles_x(angles_x)
        self.set_kicker_angles_y(angles_y)
        
    def get_corrector_angles(self):
        return np.array([node.getParam("ky") for node in self.corrector_nodes])

    def set_corrector_angles(self, angles):
        for angle, node in zip(angles, self.corrector_nodes):
            node.setParam("ky", angle)
        
    def init_part(self, x=0.0, xp=0.0, y=0.0, yp=0.0):
        self.bunch.deleteParticle(0)
        self.bunch.addParticle(x, xp, y, yp, 0.0, 0.0)
        
    def track_part(self, sublattice=0):
        self.sublattices[sublattice].trackBunch(self.bunch, self.params_dict)
        return np.array([self.bunch.x(0), self.bunch.xp(0), self.bunch.y(0), self.bunch.yp(0)])
                
    def get_trajectory(self):
        self.init_part(0.0, 0.0, 0.0, 0.0)
        coords, positions, names = [], [], []
        position_offset = 0.0
        for i, sublattice in enumerate(self.sublattices):
            if i == 1:
                position_offset = positions[-1]
            for monitor_node in self.monitor_nodes[i]:
                monitor_node.active = True
            sublattice.trackBunch(self.bunch, self.params_dict)
            for monitor_node in self.monitor_nodes[i]:
                coords.append(np.squeeze(monitor_node.data[-1]))
                positions.append(monitor_node.position + position_offset)
                names.append(monitor_node.getName().split(':')[0])
                monitor_node.clear_data()
                monitor_node.active = False

        coords = pd.DataFrame(coords, columns=['x', 'xp', 'y', 'yp'])
        coords['s'] = positions
        coords['node'] = names    
        return coords
    
    def print_status(self):
        coords_start = np.zeros(4)
        self.init_part(*coords_start)
        coords_mid = self.track_part(sublattice=0)
        coords_end = self.track_part(sublattice=1)
        for _coords, tag in zip([coords_start, coords_mid, coords_end], ['start', 'mid', 'end']):
            _coords = _coords * 1000.0
            print('Coordinates at inj_{}:'.format(tag))
            print('  x = {:.3f} [mm]'.format(_coords[0]))
            print('  y = {:.3f} [mm]'.format(_coords[2]))
            print('  xp = {:.3f} [mrad]'.format(_coords[1]))
            print('  yp = {:.3f} [mrad]'.format(_coords[3]))

    def set_inj_coords(self, coords, vcorrectors=False, **solver_kws):
        """Set the closed orbit coordinates at the injection point.
        
        This is done in four separate steps:
        
        (1) Optimize x orbit in first half (two kickers).
        (2) Optimize x orbit in first half (two kickers).
        (3) Optimize x orbit in second half (two kickers).
        (4) Optimize x orbit in second half (two kickers).
        
        In steps (1) and (2), the initial coordinates at the injection point are
        tracked backward to the start of the injection region. We then enforce
        x = x' = y = y' = 0.
        
        In steps (3) and (4), the initial coordinates at the injection point are
        tracked forward to the start of the injection region. We then enforce
        x = x' = y = y' = 0.
        
        This is done for speed/convenience and can result in terrible orbits if the 
        required coordinates at the injection point are impossible.
        """
        solver_kws.setdefault('max_nfev', 2500)
        
        if vcorrectors:
            return self.set_inj_coords_including_vcorrectors(coords, **solver_kws)
        
        x, xp, y, yp = coords
        kicker_angles_x = np.zeros(4)
        kicker_angles_y = np.zeros(4)
        
        def _evaluate(_coords):
            return 1.0e6 * np.sum(_coords**2)
        
        
        # First half of injection region   
        # ----------------------------------------------------------------------
        self.sublattices[0].reverseOrder()
        
        def _track():
            self.init_part(x, -xp, y, -yp)
            return self.track_part(sublattice=0)
        
        def cost_func(angles):
            kicker_angles_x[:2] = angles
            self.set_kicker_angles_x(kicker_angles_x)
            return _evaluate(_track()[:2])
        
        opt.least_squares(
            cost_func, 
            kicker_angles_x[:2],
            bounds=(self.min_kicker_angles_x[:2], self.max_kicker_angles_x[:2]), 
            **solver_kws
        )
        
        def cost_func(angles):
            kicker_angles_y[:2] = angles
            self.set_kicker_angles_y(kicker_angles_y)
            return _evaluate(_track()[2:])

        opt.least_squares(
            cost_func, 
            kicker_angles_y[:2],
            bounds=(self.min_kicker_angles_y[:2], self.max_kicker_angles_y[:2]),
            **solver_kws
        )         
        
        self.sublattices[0].reverseOrder()
    
    
        # Second half of injection region 
        # ----------------------------------------------------------------------
        def _track():
            self.init_part(x, +xp, y, +yp)
            return self.track_part(sublattice=1)
        
        def cost_func(angles):
            kicker_angles_x[2:] = angles
            self.set_kicker_angles_x(kicker_angles_x)
            return _evaluate(_track()[:2])

        opt.least_squares(
            cost_func, 
            kicker_angles_x[2:],
            bounds=(self.min_kicker_angles_x[2:], self.max_kicker_angles_x[2:]), 
            **solver_kws
        )     
        
        def cost_func(angles):
            kicker_angles_y[2:] = angles
            self.set_kicker_angles_y(kicker_angles_y)
            return _evaluate(_track()[2:])
        
        opt.least_squares(
            cost_func, 
            kicker_angles_y[2:],
            bounds=(self.min_kicker_angles_y[2:], self.max_kicker_angles_y[2:]),
            **solver_kws
        )     
        return self.get_kicker_angles()
        
    def set_inj_coords_including_vcorrectors(self, coords, guess=None, **solver_kws):
        self.sublattices[0].reverseOrder()
        x, xp, y, yp = coords
        kicker_angles_x = np.zeros(4)
        kicker_angles_y = np.zeros(4)
        corrector_angles_y = np.zeros(4)

        def _track_start():
            self.init_part(x, -xp, y, -yp)
            return self.track_part(sublattice=0)
        
        def _track_end():
            self.init_part(x, +xp, y, +yp)
            return self.track_part(sublattice=1)
        
        def _evaluate(_coords):
            return 1.0e6 * np.sum(_coords**2)
        
        
        # First half of injection    
        # ----------------------------------------------------------------------
        def cost_func_x(angles):
            kicker_angles_x[:2] = angles
            self.set_kicker_angles_x(kicker_angles_x)
            return _evaluate(_track_start()[:2])
        
        def cost_func_y(angles):
            kicker_angles_y[:2] = angles[0:2]
            corrector_angles_y[:2] = angles[2:4]
            self.set_kicker_angles_y(kicker_angles_y)
            self.set_corrector_angles(corrector_angles_y)
            return _evaluate(_track_start()[2:])

        opt.least_squares(
            cost_func_x, 
            kicker_angles_x[:2],
            bounds=(self.min_kicker_angles_x[:2], self.max_kicker_angles_x[:2]), 
            **solver_kws
        )         
        opt.least_squares(
            cost_func_y, 
            np.append(kicker_angles_y[:2], corrector_angles_y[:2]),
            bounds=(
                np.append(self.min_kicker_angles_y[:2], self.min_corrector_angles_y[:2]),
                np.append(self.max_kicker_angles_y[:2], self.max_corrector_angles_y[:2]),
            ), 
            **solver_kws
        )         
    
        # Second half of injection    
        # ----------------------------------------------------------------------
        def cost_func_x(angles):
            kicker_angles_x[2:] = angles
            self.set_kicker_angles_x(kicker_angles_x)
            return _evaluate(_track_end()[:2])
        
        def cost_func_y(angles):
            kicker_angles_y[2:] = angles[0:2]
            corrector_angles_y[2:] = angles[2:4]
            self.set_kicker_angles_y(kicker_angles_y)
            self.set_corrector_angles(corrector_angles_y)
            return _evaluate(_track_end()[2:])

        opt.least_squares(
            cost_func_x, 
            kicker_angles_x[2:],
            bounds=(self.min_kicker_angles_x[2:], self.max_kicker_angles_x[2:]), 
            **solver_kws
        )         
        opt.least_squares(
            cost_func_y, 
            np.append(kicker_angles_y[2:], corrector_angles_y[2:]),
            bounds=(
                np.append(self.min_kicker_angles_y[2:], self.min_corrector_angles_y[2:]),
                np.append(self.max_kicker_angles_y[2:], self.max_corrector_angles_y[2:]),
            ), 
            **solver_kws
        )         
        self.sublattices[0].reverseOrder()
        return self.get_kicker_angles()
        
    def set_inj_coords_vcorrectors(self, coords, guess=None, half='both', **solver_kws):
        self.sublattices[0].reverseOrder()
        x, xp, y, yp = coords
        corrector_angles = np.zeros(4)
        
        def _evaluate(_coords):
            return 1.0e6 * np.sum(_coords**2)
        
        if half in [0, 'both']:
                
            def cost_func(angles):
                corrector_angles[:2] = angles
                self.set_corrector_angles(corrector_angles)
                self.init_part(x, -xp, y, -yp)
                return _evaluate(self.track_part(sublattice=0)[2:])

            opt.least_squares(
                cost_func, 
                corrector_angles[:2],
                bounds=(self.min_corrector_angles_y[:2], self.max_corrector_angles_y[:2]), 
                **solver_kws
            )         

        if half in [1, 'both']:
            
            def cost_func(angles):
                corrector_angles[2:] = angles
                self.set_corrector_angles(corrector_angles)
                self.init_part(x, +xp, y, +yp)
                return _evaluate(self.track_part(sublattice=1)[2:])

            opt.least_squares(
                cost_func, 
                corrector_angles[2:],
                bounds=(self.min_corrector_angles_y[2:], self.max_corrector_angles_y[2:]), 
                **solver_kws
            )         
            self.sublattices[0].reverseOrder()

    def bias_vertical_orbit(self, **solver_kws):     
        
        # Having trouble with the cost function... how to encourage large positive y 
        # and large negative y' at the injection point. For now, just max out first 
        # two correctors and vary the last two.
        
        _angles = np.zeros(4)
        _angles[0] = self.max_corrector_angles_y[0]
        _angles[1] = self.min_corrector_angles_y[1]
        _angles = _angles
        self.set_corrector_angles(_angles)
        
        self.init_part(0.0, 0.0, 0.0, 0.0)
        x, xp, y, yp = self.track_part(sublattice=0)
       
        def cost_func(angles):
            _angles[2:] = angles
            self.set_corrector_angles(_angles)
            self.init_part(x, xp, y, yp)
            coords_end = self.track_part(sublattice=1)
            return 1.0e6 * np.sum(coords_end ** 2)
 
        solver_kws.setdefault('max_nfev', 5000)
        solver_kws.setdefault('verbose', 2)
        opt.least_squares(
            cost_func, 
            np.zeros(2), 
            bounds=(self.min_corrector_angles_y[2:], self.max_corrector_angles_y[2:]), 
            **solver_kws
        )    
        
        
        