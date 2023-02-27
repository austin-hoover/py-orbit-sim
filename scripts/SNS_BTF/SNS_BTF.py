"""SNS-BTF lattice generation."""
from __future__ import print_function
import collections
import os
import sys

import numpy as np

from orbit.py_linac.linac_parsers import SNS_LinacLatticeFactory
from orbit.py_linac.lattice.LinacApertureNodes import LinacApertureNode
from orbit.py_linac.lattice_modifications import Add_quad_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_bend_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_drift_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Replace_Quads_to_OverlappingQuads_Nodes
from orbit.py_linac.overlapping_fields.overlapping_quad_fields_lib import EngeFunction
from orbit.py_linac.overlapping_fields.overlapping_quad_fields_lib import SimpleQuadFieldFunc
from orbit.py_linac.overlapping_fields.overlapping_quad_fields_lib import PMQ_Trace3D_Function
from orbit.space_charge.sc3d import setSC3DAccNodes
from orbit.utils.xml import XmlDataAdaptor
import orbit_mpi
from spacecharge import SpaceChargeCalc3D


PIPE_DIAMETER = 0.04
SLIT_WIDTHS = {  # [mm]
    "HZ04": 0.2,
    "HZ06": 0.2,
    "VT04": 0.2,
    "VT06": 0.2,
    "HZ34a": 0.2,
    "HZ34b": 0.2,
    "VT34a": 0.2,
    "VT34b": 0.2,
    "VS06": 0.2,
    "VS06_large": 0.8
}


def file_to_dict(filename):
    """Read text file with two columns as dict."""
    dictionary = collections.OrderedDict()
    file = open(filename, "U")
    for item in file.read().split("\n"):
        if item:
            if item[0] != "#":
                try: 
                    items = item.split(", ")
                    key, value = items[0], items[1:] 
                    if len(value) == 1:
                        value = items[1] 
                    dictionary[key] = value
                except:
                    print("Skipped line '%s'" % item)
                    pass
    file.close()
    return dictionary


def quad_params_from_mstate(filename, param_name='setpoint'):
    """Load quadrupole parameters from .mstate file."""
    state_da = XmlDataAdaptor.adaptorForFile(filename)
    setpoints = collections.OrderedDict()
    for item in state_da.data_adaptors[0].data_adaptors:
        pv_name = item.getParam("setpoint_pv").split(":")
        ps_name = pv_name[1].split("_")
        mag_name = ps_name[1]
        setpoints[mag_name] = float(item.getParam(param_name))
    return setpoints


def quad_setpoints_from_mstate(filename):
    """Load quadrupole setpoints [A] from .mstate file."""
    return quad_params_from_mstate(filename, param_name='setpoint')


def quad_readbacks_from_mstate(filename):
    """Load quadrupole readbacks [T] from .mstate file."""
    return quad_params_from_mstate(filename, param_name='readback')


def get_quad_func(quad_node):
    """Generate Enge's Function for BTF quadrupoles.

    This is factory is specific to the BTF magnets. Some Enge's function parameters
    are found by fitting the measured or calculated field distributions; others are
    generated from the quadrupole length and beam pipe diameter. The parameters
    here are known for the BTF quads.

    Parameters
    ----------
    quad_node : AccNode

    Returns
    -------
    EngeFunction
    """
    name = quad_node.getName()
    if name in ["MEBT:QV02"]:
        length_param = 0.066
        acceptance_diameter_param = 0.0363
        cutoff_level = 0.001
        return EngeFunction(length_param, acceptance_diameter_param, cutoff_level)
    elif name in ["MEBT:QH01"]:
        length_param = 0.061
        acceptance_diameter_param = 0.029
        cutoff_level = 0.001
        return EngeFunction(length_param, acceptance_diameter_param, cutoff_level)
    # BTF PMQ's (arrangements of 2 pancakes per quad):
    elif name.find("FQ") >= 0:
        # number of pancakes comprising 1 quad
        npancakes = 2
        # inches to meters
        inch2meter = 0.0254
        # pole field [T] (from Menchov FEA simulation, field at inner radius 2.25 cm)
        Bpole = 0.574  # 1.2
        # inner radius (this is actually radius of inner aluminum housing, which is
        # slightly less than SmCo2 material inner radius)
        ri = 0.914 * inch2meter
        # outer radius (this is actually radius of through-holes, which is slightly
        # larger than SmCo2 material)
        ro = 1.605 * inch2meter
        # length of quad (this is length of n pancakes sancwhiched together)
        length_param = npancakes * 1.378 * inch2meter
        cutoff_level = 0.01
        return PMQ_Trace3D_Function(length_param, ri, ro, cutoff_level=cutoff_level)
    # General Enge's Function (for other quads with given aperture parameter):
    elif quad_node.hasParam("aperture"):
        length_param = quad_node.getLength()
        acceptance_diameter_param = quad_node.getParam("aperture")
        cutoff_level = 0.001
        return EngeFunction(length_param, acceptance_diameter_param, cutoff_level)
    else:
        msg = "SNS_EngeFunctionFactory Python function. "
        msg += os.linesep
        msg += "Cannot create the EngeFunction for the quad!"
        msg += os.linesep
        msg = msg + "quad name = " + quad_node.getName()
        msg = msg + os.linesep
        msg = msg + "It does not have the aperture parameter!"
        msg = msg + os.linesep
        orbitFinalize(msg)
        return None
                
    
class SNS_BTF:
    """Class to generate BTF lattice.

    Attributes
    ----------
    lattice : orbit.lattice.AccLattice
        The PyORBIT accelerator lattice instance.
    magnets : dict
        Quadrupole magnet names and strengths.
    """

    def __init__(self, coef_filename=None):
        """Constructor.

        Parameters
        ----------
        coef_filename : str
            File name for magnet coefficients.
        """
        self.coef_filename = coef_filename
        self.coeff = file_to_dict(coef_filename)
        self.lattice = None
        self.magnets = None
        self.space_charge_nodes = None
        self.aperture_nodes = None
        self.pipe_diameter = PIPE_DIAMETER
        self.slit_widths = SLIT_WIDTHS
        
    def current_to_gradient(self, quad_name, current):
        """Convert current to gradient.

        Parameters
        ----------
        quad_name : str
            Quadrupole name (i.e., 'QH01').
        current : float
            Current setpoint [A]. (Corresponds with scaled AI in IOC.)
            
        Returns
        -------
        float
            Integrated gradient [T].
        """
        current = float(current)
        sign = np.sign(current)
        current = np.abs(current)
        try:
            A = float(self.coeff[quad_name][0])
            B = float(self.coeff[quad_name][1])
            gradient = sign * (A * current + B * current**2) 
        except KeyError: 
            print(
                "Do not know conversion factor for element {}".format(quad_name),
                "gradient value not assigned.",
            ) 
            gradient = []
        return gradient
    
    def gradient_to_current(self, quad_name, gradient):
        """Convert gradient to current.

        Parameters
        ----------
        quad_name : str
            Quadrupole name (i.e., 'QH01').
        gradient : float
            Integrated gradient [T].
            
        Returns
        -------
        current : float
            Current [A]. (Corresponds with scaled AI in IOC.)
        """
        gradient = float(gradient) 
        sign = np.sign(gradient)
        gradient = np.abs(gradient)
        try:
            A = float(self.coeff[quad_name][0])
            B = float(self.coeff[quad_name][1])
            if B == 0 and A == 0:
                current = 0.0
            elif B == 0 and A != 0 :
                current = gradient / A
            else:
                current = 0.5 * (A / B) * (-1.0 + np.sqrt(1.0 + 4.0 * gradient * B / A**2))
        except KeyError: 
            print(
                "Do not know conversion factor for element {}".format(quad_name),
                "current set to 0.",
            ) 
            current = 0
        return sign * current
            
    def get_quad_gradient(self, quad_name):
        """Return integrated gradient (GL) [T].

        By definition, a focusing quad has a positive gradient. (Special treatment
        for QV02: gradient is always positive.)
        """
        quad_node = self.magnets[quad_name]['Node']
        gradient = -quad_node.getParam('dB/dr') * quad_node.getLength()
        if quad_name == 'QV02':
            gradient = -gradient
        return gradient
    
    def set_quad_gradient(self, quad_name, gradient):
        quad_node = self.magnets[quad_name]['Node']
        kappa = -gradient / quad_node.getLength()
        if quad_name == 'QV02':
            kappa = -kappa
        quad_node.setParam('dB/dr', kappa)
        
    def set_quad_current(self, quad_name, current):
        quad_node = self.magnets[quad_name]
        quad_node['current'] = current
    
    def update_quad(self, quad_name=None, value=None, value_type=None, verbose=True):
        """Update quadrupole field strength (current or gradient).
                
        Parameters
        ----------
        quad_name : str
            The name of the quadrupole element.
        value : float
            Either the quadrupole current [A] or integrated field gradient [T].
        value_type : {'current', 'gradient'}
            Determines the meaning of `value`.
        """
        if quad_name not in self.quads:
            print("Unknown quadrupole '{}'.".format(quad_name))
            return
    
        value = float(value)
        if value_type == 'current':
            current = value
            gradient = self.current_to_gradient(quad_name, current)
        elif value_type == 'gradient':
            gradient = value
            current = self.gradient_to_current(quad_name, gradient)  
            
        self.set_quad_gradient(quad_name, gradient)
        self.set_quad_current(quad_name, current)
        if verbose:
            print(
                "Updated {} to I={:.3f} [A] (GL={:.3f} [T])."
                .format(quad_name, current, gradient)
            )
            
    def update_quads_from_mstate(self, filename, value_type=None, verbose=True):
        setpoints = dict()
        if value_type == 'gradient':
            setpoints = quad_readbacks_from_mstate(filename)
        elif value_type == 'current':
            setpoints = quad_setpoints_from_mstate(filename)
        for quad_name, value in setpoints.items():
            self.update_quad(quad_name, value, value_type=value_type, verbose=verbose)
                        
    def update_pmq(self, quad_name=None, value=None, field='gradient'):
        if quad_name not in self.quads:
            print("Unknown quadrupole '{}'.".format(quad_name))
            return
        value = float(value)
        node = self.magnets[quad_name]['Node']
        if field == 'gradient':
            gradient = value
            length = node.getLength()
            kappa = -gradient / length
            self.magnets[name]['Node'].setParam('dB/dr', kappa)
            print("Updated {} to GL = {:.3f} [T] (dB/dr = {:.3f} [T/m], L = {:.3f} [m])."
                  .format(quad_name, gradient, kappa, length))
        elif field == 'length':
            length = value
            gradient = -node.getParam('dB/dr') * node.getLength()
            node.setLength(length)
            kappa = -gradient / node.getLength() 
            node.setParam('dB/dr', kappa)
            print("Updated {} to L = {:.3f} [m] (dB/dr = {:.3f} [T/m], GL = {:.3f} [T])."
                  .format(quad_name, length, kappa, gradient))
        else:
            raise TypeError("Invalid field '{}' for PMQ".format(field))
            
    def make_pmq_fields_overlap(self, z_step=0.001, verbose=True):
        pmq_names = ['MEBT:FQ{}'.format(i) for i in range(14, 33)]
        if verbose:
            for pmq_name in pmq_names:
                print('Replacing {} with overlapping field model.'.format(pmq_name))
        Replace_Quads_to_OverlappingQuads_Nodes(
            self.lattice,
            z_step,
            accSeq_Names=['MEBT3'],
            quad_Names=pmq_names,
            EngeFunctionFactory=get_quad_func,
        )
        
    def add_space_charge_nodes(
        self,
        grid_size_x=64, 
        grid_size_y=64,
        grid_size_z=64,
        path_length_min=0.01,
        n_bunches=3,
        freq=402.5e-6,
        verbose=True,
    ):
        sc_calc = SpaceChargeCalc3D(grid_size_x, grid_size_y, grid_size_z)
        if n_bunches > 1: 
            sc_calc.numExtBunches(n_bunches)
            sc_calc.freqOfBunches(freq)
        self.space_charge_nodes = setSC3DAccNodes(self.lattice, path_length_min, sc_calc)
        if verbose:
            print(
                'Added {} space charge nodes'.format(len(self.space_charge_nodes)),
                '(grid={}X{}X{}, n_bunches={}, path_length_min={})'.format(
                    grid_size_x, 
                    grid_size_y, 
                    grid_size_z, 
                    n_bunches, 
                    path_length_min,
                )
            )
        return self.space_charge_nodes
    
    def add_aperture_nodes(self, drift_step=0.1, start=0.0, stop=None, verbose=True):
        aperture_nodes = Add_quad_apertures_to_lattice(self.lattice)
        aperture_nodes = Add_bend_apertures_to_lattice(self.lattice, aperture_nodes, step=0.1)
        if stop is None:
            stop = self.lattice.getLength()
        self.aperture_nodes = Add_drift_apertures_to_lattice(
            self.lattice,
            start,
            stop,
            drift_step,
            self.pipe_diameter,
            aperture_nodes,
        )
        if verbose:
            print(
                'Added {} aperture nodes.'.format(len(self.aperture_nodes)),
                'pipe_diameter={}, drift_step={}, start={}, stop={},'.format(
                    self.pipe_diameter,
                    drift_step, 
                    start, 
                    stop, 
                )
            )
        return self.aperture_nodes

    def add_slit(self, slit_name, pos=0.0, width=None):
        """Add a slit to the lattice.

        Parameters
        ----------
        slit_name : str
            The name of slit, e.g., 'HZ04'.
        pos : float
            Transverse position of slit [mm] (bunch center is at zero).
        width : float or None
            Width of slit [mm]. If None, uses lookup table.
            
        Returns
        -------
        aperture_node : LinacApertureNode
        """
        if width is None:
            width = self.slit_widths[slit_name]
            
        if slit_name.startswith('V'):
            dx = width * 1.0e-3
            dy = 1.1 * self.pipe_diameter
            c = pos * 1.0e-3
            d = 0.0
        elif slit_name.startswith('H'):
            dy = width * 1.0e-3
            dx = 1.1 * self.pipe_diameter
            d = pos * 1.0e-3
            c = 0.0
        else:
            raise KeyError("Cannot determine plane for slit {}".format(slit_name))
            
        a = 0.5 * dx
        b = 0.5 * dy
        shape = 3  # rectangular
        slit_node = self.lattice.getNodeForName('MEBT:{}'.format(slit_name))
        pos = slit_node.getPosition()
        
        aperture_node = LinacApertureNode(
            shape,
            0.5 * dx,
            0.5 * dy,
            c=c,
            d=d,
            pos=pos,
            name=slit_name,
        )
        aperture_node.setName('{}:Aprt'.format(slit_node.getName()))
        aperture_node.setSequence(slit_node.getSequence())
        slit_node.addChildNode(aperture_node, slit_node.ENTRANCE)
        print('Inserted {} at s={:.3f} [mm].'.format(slit_name, pos))
        return aperture_node
    
    def init_lattice(self, xml=None, beamlines=None, max_drift_length=0.012, verbose=True):
        """Initialize lattice from xml file.
        
        Parameters
        ----------
        xml : str
            Path to the XML file.
        beamlines : list[str]
            List of beamlines to include in the lattice construction.
        max_drift_length : float
            Maximum drift length [m].
        """
        if xml is None:
            raise ValueError('No xml file provided.')
        if beamlines is None:
            beamlines = ["MEBT1", "MEBT2", "MEBT3"]
            
        btf_linac_factory = SNS_LinacLatticeFactory()
        btf_linac_factory.setMaxDriftLength(max_drift_length)
        self.lattice = btf_linac_factory.getLinacAccLattice(beamlines, xml)
        self.magnets = collections.OrderedDict()
        self.quads = collections.OrderedDict()
        for quad_node in self.lattice.getQuads():
            # Node name is "beamline:quad"; example: "MEBT1:QV03".
            quad_name = quad_node.getName().split(":")[1]
            self.magnets[quad_name] = dict()
            self.magnets[quad_name]['Node'] = quad_node       
            self.quads[quad_name] = quad_node
            # Record coefficients and current if applicable. (FODO quads do not have
            # set current and are caught by try loop.)
            try:
                self.magnets[quad_name]['coeff'] = self.coeff[quad_name]
                self.magnets[quad_name]['current'] = self.gradient_to_current(
                    quad_name, 
                    self.get_quad_gradient(quad_name),
                )
            except:
                if 'FQ' in quad_name:  # FODO PMQs
                    self.magnets[quad_name]['coeff'] = [0.0, 0.0]
                    self.magnets[quad_name]['current'] = 0.0
                else:
                    # Ignore other elements (not sure what these could be... probably nothing).
                    continue
        return self.lattice 