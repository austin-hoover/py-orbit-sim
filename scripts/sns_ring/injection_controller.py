# class InjectionController:    
#     def __init__(
#         self,
#         ring=None,
#         mass=None,
#         kin_energy=None,
#         inj_mid="injm1",
#         inj_start="bpm_a09",
#         inj_end="bpm_b01",
#     ):
#         self.ring = ring
#         self.mass = mass
#         self.kin_energy = kin_energy
#         self.kicker_names = [
#             "ikickh_a10",
#             "ikickv_a10",
#             "ikickh_a11",
#             "ikickv_a11",
#             "ikickv_a12",
#             "ikickh_a12",
#             "ikickv_a13",
#             "ikickh_a13",
#         ]
#         self.kicker_nodes = [ring.getNodeForName(name) for name in self.kicker_names]

#         # Maximum injection kicker angles at 1.0 GeV [mrad].
#         self.min_kicker_angles = MIN_KICKER_ANGLES
#         self.max_kicker_angles = MAX_KICKER_ANGLES

#         # Convert the maximum kicker angles from [mrad] to [rad].
#         self.min_kicker_angles = 1.0e-3 * self.min_kicker_angles
#         self.max_kicker_angles = 1.0e-3 * self.max_kicker_angles

#         # Scale the maximum kicker angles based on kinetic energy. (They are defined for
#         # the nominal SNS kinetic energy of 1.0 GeV.)
#         self.kin_energy_scale_factor = get_pc(mass, kin_energy=1.0) / get_pc(mass, kin_energy)
#         self.min_kicker_angles *= self.kin_energy_scale_factor
#         self.max_kicker_angles *= self.kin_energy_scale_factor

#         # Identify the horizontal and vertical kicker nodes. (PyORBIT does not
#         # distinguish between horizontal/vertical kickers.)
#         self.kicker_idx_x = [0, 2, 5, 7]
#         self.kicker_idx_y = [1, 3, 4, 6]
#         self.kicker_nodes_x = [self.kicker_nodes[i] for i in self.kicker_idx_x]
#         self.kicker_nodes_y = [self.kicker_nodes[i] for i in self.kicker_idx_y]
#         self.min_kicker_angles_x = self.min_kicker_angles[self.kicker_idx_x]
#         self.max_kicker_angles_x = self.max_kicker_angles[self.kicker_idx_x]
#         self.min_kicker_angles_y = self.min_kicker_angles[self.kicker_idx_y]
#         self.max_kicker_angles_y = self.max_kicker_angles[self.kicker_idx_y]

#         # Identify dipole correctors. These will be used to make a closed bump.
#         self.vcorrector_names = ["dmcv_a09", "dchv_a10", "dchv_a13", "dmcv_b01"]
#         self.vcorrector_nodes = [self.ring.getNodeForName(name) for name in self.vcorrector_names]
#         self.max_vcorrector_angle = MAX_VCORRECTOR_ANGLE  # [mrad]
#         self.max_vcorrector_angle *= 1.0e-3  # [mrad] --> [rad]
#         self.max_vcorrector_angle *= self.kin_energy_scale_factor
#         self.min_vcorrector_angle = -self.max_vcorrector_angle

#         # Create one sublattice for the first half of the injection region (before the foil)
#         # and one sublattice for the second half of the injection region (afterthe foil).
#         self.sublattice1 = self.ring.getSubLattice(
#             self.ring.getNodeIndex(self.ring.getNodeForName(inj_start)), 
#             -1,
#         )
#         self.sublattice2 = self.ring.getSubLattice(
#             self.ring.getNodeIndex(self.ring.getNodeForName(inj_mid)),
#             self.ring.getNodeIndex(self.ring.getNodeForName(inj_end)),
#         )        
#         self.sublattices = [self.sublattice1, self.sublattice2]
        
#         # Add inactive monitor nodes
#         self.monitor_nodes = []
#         for sublattice in self.sublattices:
#             self.monitor_nodes.append([])
#             node_pos_dict = sublattice.getNodePositionsDict()
#             for node in sublattice.getNodes():
#                 monitor_node = BunchCoordsNode(axis=(0, 1, 2, 3))
#                 monitor_node.position = node_pos_dict[node][0]
#                 monitor_node.active = False
#                 add_diagnostics_node_as_child(
#                     monitor_node,
#                     parent_node=node,
#                     part_index=0,
#                     place_in_part=AccActionsContainer.BEFORE,
#                 )
#                 self.monitor_nodes[-1].append(monitor_node)
                
#         # Initialize bunch for single-particle tracking.
#         self.bunch = Bunch()
#         self.bunch.mass(mass)
#         self.bunch.getSyncParticle().kinEnergy(kin_energy)
#         self.params_dict = {"lostbunch": Bunch()}  # for apertures
#         self.bunch.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
#     def scale_kicker_limits(self, factor=1.0):
#         self.min_kicker_angles = factor * self.min_kicker_angles
#         self.max_kicker_angles = factor * self.max_kicker_angles
#         self.min_kicker_angles_x = self.min_kicker_angles[self.kicker_idx_x]
#         self.max_kicker_angles_x = self.max_kicker_angles[self.kicker_idx_x]
#         self.min_kicker_angles_y = self.min_kicker_angles[self.kicker_idx_y]
#         self.max_kicker_angles_y = self.max_kicker_angles[self.kicker_idx_y]
        
#     def get_nodes(self):
#         return self.sublattice1.getNodes() + self.sublattice2.getNodes()

#     def get_kicker_angles_x(self):
#         return np.array([node.getParam("kx") for node in self.kicker_nodes_x])

#     def get_kicker_angles_y(self):
#         return np.array([node.getParam("ky") for node in self.kicker_nodes_y])

#     def get_kicker_angles(self):
#         angles = []
#         for node in self.kicker_nodes:
#             if node in self.kicker_nodes_x:
#                 angles.append(node.getParam("kx"))
#             elif node in self.kicker_nodes_y:
#                 angles.append(node.getParam("ky"))
#         return np.array(angles)

#     def set_kicker_angles_x(self, angles=None):
#         if angles is not None:
#             for angle, node in zip(angles, self.kicker_nodes_x):
#                 node.setParam("kx", angle)

#     def set_kicker_angles_y(self, angles=None):
#         if angles is not None:
#             for angle, node in zip(angles, self.kicker_nodes_y):
#                 node.setParam("ky", angle)
                
#     def set_kicker_angles(self, angles):
#         angles_x = [angles[i] for i in self.kicker_idx_x]
#         angles_y = [angles[i] for i in self.kicker_idx_y]
#         self.set_kicker_angles_x(angles_x)
#         self.set_kicker_angles_y(angles_y)
        
#     def get_vcorrector_angles(self):
#         return np.array([node.getParam("ky") for node in self.vcorrector_nodes])

#     def set_vcorrector_angles(self, angles):
#         for angle, node in zip(angles, self.vcorrector_nodes):
#             node.setParam("ky", angle)
        
#     def init_part(self, x=0.0, xp=0.0, y=0.0, yp=0.0):
#         self.bunch.deleteParticle(0)
#         self.bunch.addParticle(x, xp, y, yp, 0.0, 0.0)
        
#     def track_part(self, sublattice=0):
#         self.sublattices[sublattice].trackBunch(self.bunch, self.params_dict)
#         return np.array([self.bunch.x(0), self.bunch.xp(0), self.bunch.y(0), self.bunch.yp(0)])
                            
#     def set_inj_coords(self, coords, guess=None, **kws):
#         coords = np.array(coords)
#         if guess is None:
#             guess = np.zeros(8)
        
#         def magnitude(_coords):
#             return 1.0e4 * np.sum(_coords**2)
                            
#         def cost_func(angles):
#             self.set_kicker_angles(angles)
#             self.init_part(0.0, 0.0, 0.0, 0.0)
#             coords_mid = self.track_part(sublattice=0)
#             coords_end = self.track_part(sublattice=1)
#             ftsr = 4 * "{:.3f} " + "| " + 4 * "{:.3f} "
#             print(
#                 ftsr.format(
#                     1000.0 * coords_mid[0], 
#                     1000.0 * coords_mid[1], 
#                     1000.0 * coords_mid[2], 
#                     1000.0 * coords_mid[3],
#                     1000.0 * coords_end[0], 
#                     1000.0 * coords_end[1], 
#                     1000.0 * coords_end[2], 
#                     1000.0 * coords_end[3],
#                 )
#             )
#             return magnitude(coords_mid - coords) + magnitude(coords_end)

#         bounds = (self.min_kicker_angles, self.max_kicker_angles)
#         opt.least_squares(cost_func, guess, bounds=bounds, **kws)
#         return self.get_kicker_angles()
    
#     def set_inj_coords_x(self, coords, **kws):
#         coords = np.array(coords)
        
#         def magnitude(_coords):
#             return 1.0e4 * np.sum(_coords**2)
                            
#         def cost_func(angles):
#             self.set_kicker_angles_x(angles)
#             self.init_part(0.0, 0.0, 0.0, 0.0)
#             coords_mid = self.track_part(sublattice=0)
#             coords_end = self.track_part(sublattice=1)
#             print(
#                 (4 * "{:.3f} " + "| " + 4 * "{:.3f} ").format(
#                     1000.0 * coords_mid[0], 
#                     1000.0 * coords_mid[1], 
#                     1000.0 * coords_mid[2], 
#                     1000.0 * coords_mid[3],
#                     1000.0 * coords_end[0], 
#                     1000.0 * coords_end[1], 
#                     1000.0 * coords_end[2], 
#                     1000.0 * coords_end[3],
#                 )
#             )
#             return magnitude(coords_mid[:2] - coords[:2]) + magnitude(coords_end[:2])

#         bounds = (self.min_kicker_angles_x, self.max_kicker_angles_x)
#         opt.least_squares(cost_func, np.zeros(4), bounds=bounds, **kws)

#     def set_inj_coords_y(self, coords, **kws):
#         coords = np.array(coords)
        
#         def magnitude(_coords):
#             return 1.0e4 * np.sum(_coords**2)
                            
#         def cost_func(angles):
#             self.set_kicker_angles_y(angles)
#             self.init_part(0.0, 0.0, 0.0, 0.0)
#             coords_mid = self.track_part(sublattice=0)
#             coords_end = self.track_part(sublattice=1)
#             print(
#                 (4 * "{:.3f} " + "| " + 4 * "{:.3f} ").format(
#                     1000.0 * coords_mid[0], 
#                     1000.0 * coords_mid[1], 
#                     1000.0 * coords_mid[2], 
#                     1000.0 * coords_mid[3],
#                     1000.0 * coords_end[0], 
#                     1000.0 * coords_end[1], 
#                     1000.0 * coords_end[2], 
#                     1000.0 * coords_end[3],
#                 )
#             )
#             return magnitude(coords_mid[2:] - coords[2:]) + magnitude(coords_end[2:])

#         bounds = (self.min_kicker_angles_y, self.max_kicker_angles_y)
#         opt.least_squares(cost_func, np.zeros(4), bounds=bounds, **kws)
        
#     def set_inj_coords_x_halves(self, coords, **kws):
#         coords = np.array(coords)
#         kicker_angles = np.zeros(4)
#         lb = self.min_kicker_angles_x
#         ub = self.max_kicker_angles_x
        
#         def magnitude(_coords):
#             return 1.0e4 * np.sum(_coords**2)
        
#         def cost_func(angles):
#             kicker_angles[:2] = angles
#             self.set_kicker_angles_x(kicker_angles)
#             self.init_part(0.0, 0.0, 0.0, 0.0)
#             coords_mid = self.track_part(sublattice=0)
#             print(
#                 (4 * "{:.3f} ").format(
#                     1000.0 * coords_mid[0], 
#                     1000.0 * coords_mid[1], 
#                     1000.0 * coords_mid[2], 
#                     1000.0 * coords_mid[3],
#                 )
#             )
#             return magnitude(coords_mid[:2] - coords[:2])
                            
#         opt.least_squares(cost_func, np.zeros(2), bounds=(lb[:2], ub[:2]), **kws)
                
#         def cost_func(angles):
#             kicker_angles[2:] = angles
#             self.set_kicker_angles_x(kicker_angles)
#             self.init_part(0.0, 0.0, 0.0, 0.0)
#             coords_mid = self.track_part(sublattice=0)
#             coords_end = self.track_part(sublattice=1)
#             print(
#                 (4 * "{:.3f} " + "| " + 4 * "{:.3f} ").format(
#                     1000.0 * coords_mid[0], 
#                     1000.0 * coords_mid[1], 
#                     1000.0 * coords_mid[2], 
#                     1000.0 * coords_mid[3],
#                     1000.0 * coords_end[0], 
#                     1000.0 * coords_end[1], 
#                     1000.0 * coords_end[2], 
#                     1000.0 * coords_end[3],
#                 )
#             )
#             return magnitude(coords_end[:2])

#         opt.least_squares(cost_func, np.zeros(2), bounds=(lb[2:], ub[2:]), **kws)

#     def set_inj_coords_y_halves(self, coords, **kws):
#         coords = np.array(coords)
#         kicker_angles = np.zeros(4)
#         lb = self.min_kicker_angles_y
#         ub = self.max_kicker_angles_y
        
#         self.sublattices[0].reverseOrder()
        
#         def magnitude(_coords):
#             return 1.0e4 * np.sum(_coords**2)
        
#         def cost_func(angles):
#             kicker_angles[:2] = angles
#             self.set_kicker_angles_y(kicker_angles)
#             self.init_part(coords[0], -coords[1], coords[2], -coords[3])
#             coords_out = self.track_part(sublattice=0)
#             print(
#                 (4 * "{:.3f} ").format(
#                     1000.0 * coords_out[0], 
#                     1000.0 * coords_out[1], 
#                     1000.0 * coords_out[2], 
#                     1000.0 * coords_out[3],
#                 )
#             )
#             return magnitude(coords_out[2:])
                            
#         opt.least_squares(cost_func, np.zeros(2), bounds=(lb[:2], ub[:2]), **kws)
        
#         self.sublattices[0].reverseOrder()
#         self.init_part(0.0, 0.0, 0.0, 0.0)
#         coords = self.track_part(sublattice=0)
                
#         def cost_func(angles):
#             kicker_angles[2:] = angles
#             self.set_kicker_angles_y(kicker_angles)
#             self.init_part(coords[0], coords[1], coords[2], coords[3])
#             coords_out = self.track_part(sublattice=1)
#             print(
#                 (4 * "{:.3f} ").format(
#                     1000.0 * coords_out[0], 
#                     1000.0 * coords_out[1], 
#                     1000.0 * coords_out[2], 
#                     1000.0 * coords_out[3],
#                 )
#             )
#             return magnitude(coords_out[2:])

#         opt.least_squares(cost_func, np.zeros(2), bounds=(lb[2:], ub[2:]), **kws)

#     def set_inj_coords_fast(self, coords, vcorrectors=False, **solver_kws):
#         solver_kws.setdefault("max_nfev", 5000)   
#         solver_kws.setdefault("verbose", 2)   
#         x, xp, y, yp = coords
#         kicker_angles_x = np.zeros(4)
#         kicker_angles_y = np.zeros(4)
        
#         def _evaluate(_coords):
#             return 1.0e6 * np.sum(_coords**2)
        
        
#         # First half of injection region   

#         self.sublattices[0].reverseOrder()
        
#         def _track():
#             self.init_part(x, -xp, y, -yp)
#             return self.track_part(sublattice=0)
        
#         def cost_func(angles):
#             kicker_angles_x[:2] = angles
#             self.set_kicker_angles_x(kicker_angles_x)
#             return _evaluate(_track()[:2])
        
#         opt.least_squares(
#             cost_func, 
#             kicker_angles_x[:2],
#             bounds=(self.min_kicker_angles_x[:2], self.max_kicker_angles_x[:2]), 
#             **solver_kws
#         )
        
#         def cost_func(angles):
#             kicker_angles_y[:2] = angles
#             self.set_kicker_angles_y(kicker_angles_y)
#             return _evaluate(_track()[2:])

#         opt.least_squares(
#             cost_func, 
#             kicker_angles_y[:2],
#             bounds=(self.min_kicker_angles_y[:2], self.max_kicker_angles_y[:2]),
#             **solver_kws
#         )         
        
#         self.sublattices[0].reverseOrder()
    
    
#         # Second half of injection region 

#         def _track():
#             self.init_part(x, +xp, y, +yp)
#             return self.track_part(sublattice=1)
        
#         def cost_func(angles):
#             kicker_angles_x[2:] = angles
#             self.set_kicker_angles_x(kicker_angles_x)
#             return _evaluate(_track()[:2])

#         opt.least_squares(
#             cost_func, 
#             kicker_angles_x[2:],
#             bounds=(self.min_kicker_angles_x[2:], self.max_kicker_angles_x[2:]), 
#             **solver_kws
#         )     
        
#         def cost_func(angles):
#             kicker_angles_y[2:] = angles
#             self.set_kicker_angles_y(kicker_angles_y)
#             return _evaluate(_track()[2:])
        
#         opt.least_squares(
#             cost_func, 
#             kicker_angles_y[2:],
#             bounds=(self.min_kicker_angles_y[2:], self.max_kicker_angles_y[2:]),
#             **solver_kws
#         )     
#         return self.get_kicker_angles()

#     def set_inj_coords_vcorrectors(self, coords, **solver_kws):
#         solver_kws.setdefault("max_nfev", 5000)   
#         solver_kws.setdefault("verbose", 2)   
#         coords = np.array(coords)
#         x, xp, y, yp = coords
        
#         def magnitude(_coords):
#             return 1.0e4 * np.sum(_coords**2)
                        
#         def cost_func(angles):
#             self.set_vcorrector_angles(angles)
#             self.init_part(0.0, 0.0, 0.0, 0.0)
#             coords_mid = self.track_part(sublattice=0)
#             coords_end = self.track_part(sublattice=1)
#             print(
#                 (4 * "{:.3f} " + "| " + 4 * "{:.3f} ").format(
#                     1000.0 * coords_mid[0], 
#                     1000.0 * coords_mid[1], 
#                     1000.0 * coords_mid[2], 
#                     1000.0 * coords_mid[3],
#                     1000.0 * coords_end[0], 
#                     1000.0 * coords_end[1], 
#                     1000.0 * coords_end[2], 
#                     1000.0 * coords_end[3],
#                 )
#             )
#             cost = 0.0
#             cost += magnitude(coords_mid - coords)
#             cost += magnitude(coords_end)
#             return cost
        
#         opt.least_squares(
#             cost_func, 
#             np.zeros(4),
#             bounds=(self.min_vcorrector_angle, self.max_vcorrector_angle), 
#             **solver_kws
#         )
#         return self.get_vcorrector_angles()
    
#     def get_trajectory(self):
#         self.init_part(0.0, 0.0, 0.0, 0.0)
#         coords, positions, names = [], [], []
#         position_offset = 0.0
#         for i, sublattice in enumerate(self.sublattices):
#             if i == 1:
#                 position_offset = positions[-1]
#             for monitor_node in self.monitor_nodes[i]:
#                 monitor_node.active = True
#             sublattice.trackBunch(self.bunch, self.params_dict)
#             for monitor_node in self.monitor_nodes[i]:
#                 coords.append(np.squeeze(monitor_node.data[-1]))
#                 positions.append(monitor_node.position + position_offset)
#                 names.append(monitor_node.getName().split(":")[0])
#                 monitor_node.clear_data()
#                 monitor_node.active = False
#         coords = pd.DataFrame(coords, columns=["x", "xp", "y", "yp"])
#         coords["s"] = positions
#         coords["node"] = names    
#         return coords
    
#     def print_inj_coords(self):
#         coords_start = np.zeros(4)
#         self.init_part(*coords_start)
#         coords_mid = self.track_part(sublattice=0)
#         coords_end = self.track_part(sublattice=1)
#         for _coords, tag in zip([coords_start, coords_mid, coords_end], ["start", "mid", "end"]):
#             _coords = _coords * 1000.0
#             print("Coordinates at inj_{}:".format(tag))
#             print("  x = {:.3f} [mm]".format(_coords[0]))
#             print("  y = {:.3f} [mm]".format(_coords[2]))
#             print("  xp = {:.3f} [mrad]".format(_coords[1]))
#             print("  yp = {:.3f} [mrad]".format(_coords[3]))