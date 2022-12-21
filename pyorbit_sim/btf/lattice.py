"""BTF lattice generation."""
from __future__ import print_function
import collections
import os
import sys

import numpy as np

from orbit.py_linac.linac_parsers import SNS_LinacLatticeFactory
from orbit.py_linac.lattice.LinacApertureNodes import LinacApertureNode
from orbit.py_linac.overlapping_fields.overlapping_quad_fields_lib import EngeFunction
from orbit.py_linac.overlapping_fields.overlapping_quad_fields_lib import SimpleQuadFieldFunc
from orbit.py_linac.overlapping_fields.overlapping_quad_fields_lib import PMQ_Trace3D_Function
from orbit.utils.xml import XmlDataAdaptor


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
        # Get magnet name from pvname.
        pvname = item.getParam("setpoint_pv").split(":")
        psname = pvname[1].split("_")
        magname = psname[1]
        setpoints[magname] = float(item.getParam(param_name))
    return setpoints


def quad_setpoints_from_mstate(filename):
    """Load quadrupole setpoints [A] from .mstate file."""
    return quad_params_from_mstate(filename, 'setpoint')


def quad_readbacks_from_mstate(filename):
    """Load quadrupole readbacks [T] from .mstate file."""
    return quad_params_from_mstate(filename, 'readback')


def get_quad_func(quad):
    """Generate Enge's Function for BTF quadrupoles.

    This is factory is specific to the BTF magnets. Some Enge's function parameters
    are found by fitting the measured or calculated field distributions; others are
    generated from the quadrupole length and beam pipe diameter. The parameters
    here are known for the BTF quads.

    Parameters
    ----------
    quad : AccNode

    Returns
    -------
    EngeFunction
    """
    name = quad.getName()
    if name in ["MEBT:QV02"]:
        length_param = 0.066
        acceptance_diameter_param = 0.0363
        cutoff_level = 0.001
        func = EngeFunction(length_param, acceptance_diameter_param, cutoff_level)
        return func
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
    elif quad.hasParam("aperture"):
        length_param = quad.getLength()
        acceptance_diameter_param = quad.getParam("aperture")
        cutoff_level = 0.001
        return EngeFunction(length_param, acceptance_diameter_param, cutoff_level)
    else:
        msg = "SNS_EngeFunctionFactory Python function. "
        msg += os.linesep
        msg += "Cannot create the EngeFunction for the quad!"
        msg += os.linesep
        msg = msg + "quad name = " + quad.getName()
        msg = msg + os.linesep
        msg = msg + "It does not have the aperture parameter!"
        msg = msg + os.linesep
        orbitFinalize(msg)
        return None
    
    
class MagnetConverter:
    """Convert magnet gradient/current."""

    def __init__(self, coef_filename=None):
        self.coef_filename = coef_filename
        self.coeff = file_to_dict(self.coef_filename)

    def c2gl(self, quadname, scaledAI):
        """Convert current to gradient.

        quadname : str
            Quadrupole name (i.e., 'QH01').
        scaledAI : float
            Current setpoint (corresponds with scaled AI in IOC) [A].
        """
        scaledAI = float(scaledAI)
        try:
            A = float(self.coeff[quadname][0])
            B = float(self.coeff[quadname][1])
            GL = A * scaledAI + B * scaledAI**2
        except KeyError:
            print(
                "Do not know conversion factor for element {};".format(quadname),
                "gradient value not assigned.",
            )
            GL = []
        return GL

    def gl2c(self, quadname, GL):
        """Convert gradient to current.

        quadname : str
            Quadrupole name (i.e., 'QH01').
        GL : float
            Integrated gradient [T].
        """
        GL = float(GL)
        try:
            A = float(self.coeff[quadname][0])
            B = float(self.coeff[quadname][1])
            if B == 0 and A == 0:  # handle case of 0 coefficients
                scaledAI = 0
            elif B == 0 and A != 0:  # avoid division by 0 for quads with 0 offset
                scaledAI = GL / A
            else:
                scaledAI = 0.5 * (A / B) * (-1 + np.sqrt(1 + 4 * GL * B / A**2))
        except KeyError:
            print(
                "Do not know conversion factor for element {};".format(quadname),
                "current set to zero.",
            )
            scaledAI = 0
        return scaledAI

    def igrad2current(self, inputdict):
        """inputdict has key = magnet name, value = integrated gradient GL [T]."""
        outputdict = OrderedDict.fromkeys(self.coeff.keys(), [])
        for name in inputdict:
            try:
                outputdict[name] = self.gl2c(name, inputdict[name])
            except:
                print("Something went wrong on element {}.".format(name))
        return outputdict

    def current2igrad(self, inputdict):
        """inputdict has key = magnet name, value = current setpoint [A]."""
        outputdict = OrderedDict.fromkeys(self.coeff.keys(), [])
        for name in inputdict:
            try:
                outputdict[name] = self.c2gl(name, inputdict[name])
            except:
                print("Something went wrong on element {}.".format(name))
        return outputdict


class BTFLatticeGenerator(MagnetConverter):
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
        MagnetConverter.__init__(self, coef_filename=coef_filename)
        self.lattice = None
        self.magnets = None
        self.pipe_diameter = PIPE_DIAMETER
        self.slit_widths = SLIT_WIDTHS
        
    def init_lattice(self, xml=None, beamlines=None, max_drift_length=0.012):
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
        for quad in self.lattice.getQuads():
            # Node name is "beamline:quad"; example: "MEBT1:QV03".
            quad_name = quad.getName().split(":")[1]
            self.magnets[quad_name] = dict()
            self.magnets[quad_name]["Node"] = quad
            # By convention, focusing quad has GL > 0. QV02 is always positive.
            GL = -quad.getParam("dB/dr") * quad.getLength()
            if quad_name == "QV02":
                GL = -GL
            # Record coefficients and current if applicable (FODO quads do not have
            # set current and are caught by try loop.)
            try:
                self.magnets[quad_name]["coeff"] = self.coeff[quad_name]
                self.magnets[quad_name]["current"] = self.gl2c(quad_name, GL)
            except:
                # Catch quads that do not have PV names.
                if "FQ" in quad_name:  # FODO PMQs
                    self.magnets[quad_name]["coeff"] = [0, 0]
                    self.magnets[quad_name]["current"] = 0
                else:
                    # Ignore other elements (not sure what these could be... probably nothing).
                    continue
        return self.lattice

    def update_quads(self, units="Amps", **setpoints):
        """Update quadrupole gradients in lattice definition.

        units : str
            The units of the values in `setpoints`. 
        **setpoints
            Keys are quadrupole names; values are currents. Names should not include
            beamline name ('QH01' instead of 'MEBT1:QH01').
        """
        for element_name, value in setpoints.items():
            if units == "Amps":
                GL = self.c2gl(element_name, float(value))
                newcurrent = float(value)
            elif units == "Tesla":
                GL = float(value)
                newcurrent = self.gl2c(element_name, float(value))
            else:
                raise (
                    TypeError,
                    "Do not understand unit {} for quadrupole setting".format(units),
                )
            try:
                self.magnets[element_name]["current"] = newcurrent
                # Update gradient in node definition. By convention, the
                # focusing quad has GL > 0. (Special treatment for QV02 polarity:
                # kappa + current, GL are always positive.)
                newkappa = -GL / self.magnets[element_name]["Node"].getLength()
                if element_name == "QV02":
                    newkappa = -newkappa
                self.magnets[element_name]["Node"].setParam("dB/dr", newkappa)
                print(
                    "Changed {} to {:.3f} [A] (dB/dr={:.3f} [T/m], GL={:.3f} [T]).".format(
                        element_name, float(newcurrent), newkappa, GL
                    )
                )
            except KeyError:
                print("Element {} is not defined.".format(element_name))

    def load_quads(self, filename, units="Tesla"):
        """Update quadrupoles from .mstate file."""
        if filename.endswith(".mstate"):
            if units == "Tesla":
                setpoints = quad_readbacks_from_mstate(filename)
            elif units == "Amps":
                setpoints = quad_setpoints_from_mstate(filename)
            self.update_quads(units=units, **setpoints)
        else:
            raise NameError("{} lacks .mstate extension.".format(filename))

    def update_pmqs(self, field='GL', **setpoints):
        """Update quadrupole gradients in lattice definition.
        
        field : str
            The field to edit. 
        **setpoints
            Key is quadrupole name ('FQ01'); value is magnetic field strength [Tesla].         
        """
        for element_name, value in setpoints.items():
            if field == "GL":
                newGL = float(value)
                try:
                    L = self.magnets[element_name]["Node"].getLength()
                    newkappa = newGL / L  # convention: focusing quad has + GL
                    self.magnets[element_name]["Node"].setParam("dB/dr", newkappa)
                    print(
                        "Changed %s to GL = %.3f T (dB/dr=%.3f T/m, L=%.3f)"
                        % (element_name, float(newGL), newkappa, L)
                    )
                except KeyError:
                    print("Element %s is not defined" % (element_name))
            elif field == "Length":
                newL = float(value)
                try:
                    GL = (
                        self.magnets[element_name]["Node"].getParam("dB/dr")
                        * self.magnets[element_name]["Node"].getLength()
                    )
                    self.magnets[element_name]["Node"].setLength(newL)
                    # changing length but holding GL fixed changes effective strength kappa
                    newkappa = GL / self.magnets[element_name]["Node"].getLength()
                    self.magnets[element_name]["Node"].setParam("dB/dr", newkappa)
                    print(
                        "Changed %s to L = %.3f m (dB/dr=%.3f T/m, GL=%.3f T)"
                        % (element_name, float(newL), newkappa, GL)
                    )
                except KeyError:
                    print("Element {} is not defined".format(element_name))
            else:
                raise (TypeError, "Do not understand field={} for PMQ element".format(field))

    def add_slit(self, slit_name, pos=0.0, width=None):
        """Add a slit to the lattice.

        Parameters
        ----------
        slit_name : str
            The name of slit, e.g., 'MEBT:HZ04'.
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

        # Determine if horizontal or vertical slit.
        if slit_name[0] == "V":
            dx = width * 1e-3
            dy = 1.1 * self.pipe_diameter
            c = pos * 1e-3
            d = 0.0
        elif slit_name[0] == "H":
            dy = width * 1e-3
            dx = 1.1 * self.pipe_diameter
            d = pos * 1e-3
            c = 0.0
        else:
            raise KeyError("Cannot determine plane for slit {}".format(slit_name))

        a = 0.5 * dx
        b = 0.5 * dy
        shape = 3  # rectangular

        # Create aperture node. In this call, pos is longitudinal position.
        slit_node = self.lattice.getNodeForName("MEBT:" + slit_name)
        aperture_node = LinacApertureNode(
            shape,
            a,
            b,
            c=c,
            d=d,
            pos=slit_node.getPosition(),
            name=slit_name,
        )

        # Add as child to slit marker node.
        aperture_node.setName(slit_node.getName() + ":Aprt")
        aperture_node.setSequence(slit_node.getSequence())
        slit_node.addChildNode(aperture_node, slit_node.ENTRANCE)
        print("Inserted {} at {:.3f} mm".format(slit_name, pos))

        return aperture_node
