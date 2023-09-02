"""SNS Beam Test Facility (BTF)."""
from __future__ import print_function
import collections
import copy
import os
import sys
import warnings

import numpy as np

from bunch import Bunch
from bunch import BunchTwissAnalysis
from orbit.envelope import DanilovEnvelope20
from orbit.envelope import DanilovEnvelopeSolverNode20
from orbit.envelope import set_danilov_envelope_solver_nodes_20
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.py_linac.lattice import AxisFieldRF_Gap
from orbit.py_linac.lattice import AxisField_and_Quad_RF_Gap
from orbit.py_linac.lattice import BaseRF_Gap
from orbit.py_linac.lattice import Bend
from orbit.py_linac.lattice import Drift
from orbit.py_linac.lattice import LinacApertureNode
from orbit.py_linac.lattice import LinacEnergyApertureNode
from orbit.py_linac.lattice import LinacPhaseApertureNode
from orbit.py_linac.lattice import OverlappingQuadsNode
from orbit.py_linac.lattice import Quad
from orbit.py_linac.lattice_modifications import Add_bend_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_drift_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_quad_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_rfgap_apertures_to_lattice
from orbit.py_linac.lattice_modifications import AddMEBTChopperPlatesAperturesToSNS_Lattice
from orbit.py_linac.lattice_modifications import AddScrapersAperturesToLattice
from orbit.py_linac.lattice_modifications import GetLostDistributionArr
from orbit.py_linac.lattice_modifications import Replace_BaseRF_Gap_and_Quads_to_Overlapping_Nodes
from orbit.py_linac.lattice_modifications import Replace_BaseRF_Gap_to_AxisField_Nodes
from orbit.py_linac.lattice_modifications import Replace_Quads_to_OverlappingQuads_Nodes
from orbit.py_linac.linac_parsers import SNS_LinacLatticeFactory
from orbit.py_linac.overlapping_fields.overlapping_quad_fields_lib import EngeFunction
from orbit.py_linac.overlapping_fields.overlapping_quad_fields_lib import PMQ_Trace3D_Function
from orbit.py_linac.overlapping_fields.overlapping_quad_fields_lib import SimpleQuadFieldFunc
from orbit.space_charge.sc3d import setSC3DAccNodes
from orbit.space_charge.sc3d import setUniformEllipsesSCAccNodes
from orbit.utils import consts
from orbit.utils.xml import XmlDataAdaptor
import orbit_mpi
from orbit_mpi import mpi_datatype
from orbit_mpi import mpi_op
from spacecharge import SpaceChargeCalc3D
from spacecharge import SpaceChargeCalcUnifEllipse


PIPE_DIAMETER = 0.040  # [m]
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
    "VS06_large": 0.8,
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


def quad_params_from_mstate(filename, param_name="setpoint"):
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
    return quad_params_from_mstate(filename, param_name="setpoint")


def quad_readbacks_from_mstate(filename):
    """Load quadrupole readbacks [T] from .mstate file."""
    return quad_params_from_mstate(filename, param_name="readback")


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
        n_pancakes = 2
        inch2meter = 0.0254
        # Pole field [T] (from Menchov FEA simulation, field at inner radius 2.25 cm).
        Bpole = 0.574  # 1.2
        # Inner radius -- this is actually radius of inner aluminum housing, which is
        # slightly less than SmCo2 material inner radius).
        ri = 0.914 * inch2meter
        # Outer radius -- this is actually radius of through-holes, which is slightly
        # larger than SmCo2 material).
        ro = 1.605 * inch2meter
        # Length of quad (this is length of n pancakes sandwiched together).
        length_param = n_pancakes * 1.378 * inch2meter
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
    magnets : OrderedDict
        Keys are quadrupole magnet names without sequence prefix ("QV02"
        instead of "MEBT:QV02". Each value is a dict with the following
        keys:
        - "current": magnet current [A]
        - "coeff": A and B coefficients for gradient-current conversion.
        - "node": the corresponding AccNode.
    quad_names : list
        List of quadrupole names ("MEBT:QV02").
    quad_names_fodo : list
        List of FODO quadrupole names.
    quad_names_no_fodo : list
        List of non-FODo quadrupole names.
    """

    def __init__(self, coef_filename=None, rf_frequency=402.5e+06):
        """Constructor.

        Parameters
        ----------
        coef_filename : str
            File name for magnet coefficients.
        """
        self.coef_filename = coef_filename
        self.coeff = file_to_dict(coef_filename)
        self.lattice = None
        self.magnets = collections.OrderedDict()
        self.quad_names = []
        self.quad_names_fodo = []
        self.quad_names_no_fodo = []
        self.space_charge_nodes = None
        self.aperture_nodes = None
        self.pipe_diameter = PIPE_DIAMETER
        self.slit_widths = SLIT_WIDTHS
        self.rf_frequency = rf_frequency
        
    def shorten_quad_name(self, name):
        """MEBT:QH01 -> QH01."""
        if name.startswith("MEBT"):
            return name.split(":")[1]
        return name
        
    def init_lattice(
        self, 
        xml_filename=None, 
        sequences=None, 
        max_drift_length=0.012, 
        verbose=True
    ):
        """Initialize lattice from XML file.

        Parameters
        ----------
        xml_filename : str
            XML file name.
        sequences : list[str]
            List of sequences to include in the lattice construction. ("MEBT1",
            "MEBT2", "MEBT3").
        max_drift_length : float
            Maximum drift length [m].
        """
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

        if xml_filename is None:
            raise ValueError("No xml file provided.")
        if sequences is None:
            sequences = ["MEBT1", "MEBT2", "MEBT3"]
            
        lattice_factory = SNS_LinacLatticeFactory()
        lattice_factory.setMaxDriftLength(max_drift_length)
        self.lattice = lattice_factory.getLinacAccLattice(sequences, xml_filename)
        
        self.quad_names = []
        self.quad_names_fodo = []
        self.quad_names_no_fodo = []
        for node in self.lattice.getQuads():
            self.quad_names.append(node.getName())
            if "FQ" in node.getName():
                self.quad_names_fodo.append(node.getName())
            else:
                self.quad_names_no_fodo.append(node.getName())
        self.quad_names_short = [self.shorten_quad_name(name) for name in self.quad_names]
        self.quad_names_fodo_short = [self.shorten_quad_name(name) for name in self.quad_names_fodo]
        self.quad_names_no_fodo_short = [self.shorten_quad_name(name) for name in self.quad_names_no_fodo]
        
        for node in self.lattice.getQuads():
            name = node.getName()
            name_short = self.shorten_quad_name(name)
            self.magnets[name_short] = dict()
            self.magnets[name_short]["node"] = node
            if name in self.quad_names_fodo:
                self.magnets[name_short]["coeff"] = [0.0, 0.0]
            elif name_short in self.coeff:
                coeff = self.coeff[name_short]
                gradient = self.get_quad_gradient(name_short)
                current = self.quad_gradient_to_current(name_short, gradient)
                self.magnets[name_short]["coeff"] = coeff
                print("{}: I={:.4f}, coeff={}".format(name, current, coeff))
            else:
                warnings.warn("WARNING: '{}' not in coeff dict".format(name_short))   
        
        # Make sure quad currents are within limits.
        for name in self.quad_names_no_fodo_short:
            current = self.get_quad_current(name)
            min_current, max_current = self.get_quad_current_limits(name)
            if not min_current <= current <= max_current:
                print("{} current {:.3f} outside limits".format(name, current))
                current = np.clip(current, min_current, max_current)
                self.set_quad_current(name, current, verbose=True)
        
        return self.lattice
                    
    def quad_current_to_gradient(self, quad_name, current):
        """Convert current to gradient.

        Parameters
        ----------
        quad_name : str
            Quadrupole name ("QH01" or "MEBT:QH01").
        current : float
            Current setpoint [A]. (Corresponds with scaled AI in IOC.)

        Returns
        -------
        float
            Integrated gradient [T].
        """
        quad_name = self.shorten_quad_name(quad_name)
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

    def quad_gradient_to_current(self, quad_name, gradient):
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
        quad_name = self.shorten_quad_name(quad_name)
        gradient = float(gradient)
        sign = np.sign(gradient)
        gradient = np.abs(gradient)
        try:
            A = float(self.coeff[quad_name][0])
            B = float(self.coeff[quad_name][1])
            if B == 0 and A == 0:
                current = 0.0
            elif B == 0 and A != 0:
                current = gradient / A
            else:
                current = (
                    0.5 * (A / B) * (-1.0 + np.sqrt(1.0 + 4.0 * gradient * B / A**2))
                )
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
        quad_name = self.shorten_quad_name(quad_name)
        quad_node = self.magnets[quad_name]["node"]
        gradient = -quad_node.getParam("dB/dr") * quad_node.getLength()
        if quad_name == "QV02":
            gradient = -gradient
        return gradient

    def set_quad_gradient(self, quad_name, gradient, verbose=True):
        """Set quadrupole integrated gradient (GL) [T]."""
        current = self.quad_gradient_to_current(quad_name, gradient)
        quad_name = self.shorten_quad_name(quad_name)
        quad_node = self.magnets[quad_name]["node"]
        kappa = -gradient / quad_node.getLength()            
        if quad_name == "QV02":
            kappa = -kappa
        quad_node.setParam("dB/dr", kappa)
        if verbose:
            print(
                "Set {} dB/dr={} (I={:.3f}, GL={:.3f})".format(
                    quad_name, kappa, current, gradient
                )
            )
        
    def get_quad_current(self, quad_name):
        """Return quad current [A]."""
        return self.quad_gradient_to_current(quad_name, self.get_quad_gradient(quad_name))

    def set_quad_current(self, quad_name, current, verbose=True):
        gradient = self.quad_current_to_gradient(quad_name, current)
        return self.set_quad_gradient(quad_name, gradient, verbose=verbose)
        
    def get_quad_current_limits(self, quad_name):
        """Return (min, max) quad current [A].
        
        Provided by Sasha Aleksandrov (2023-08-08).
        """
        quad_name = self.shorten_quad_name(quad_name)

        # Define max absolute current.
        abs_max_current = 12.0
        if quad_name.upper() == "QH01":
            abs_max_current = 350.0
        elif quad_name.upper() == "QV02":
            abs_max_current = 400.0
            
        min_current = 0.0
        max_current = abs_max_current

        # Determine sign.
        sign = np.sign(self.get_quad_current(quad_name))
        flip = False
        if sign < 0:
            min_current = -abs_max_current
            max_current = 0.0
        elif sign == 0:
            if quad_name.upper().startswith("QV"):
                min_current = -abs_max_current
                max_current = 0.0
    
        return (min_current, max_current)
    
    def get_quad_kappa_limits(self, quad_name):
        """Return (min, max) kappa=dB/dr.
        
        These are based on current limits.
        """
        min_current, max_current = self.get_quad_current_limits(quad_name)
        node = self.lattice.getNodeForName(quad_name)
        # kappa = copy.copy(node.getParam("dB/dr"))
        kappa = node.getParam("dB/dr")
        self.set_quad_current(quad_name, min_current, verbose=False)
        kappa_min = node.getParam("dB/dr")
        self.set_quad_current(quad_name, max_current, verbose=False)
        kappa_max = node.getParam("dB/dr")
        if kappa_max < kappa_min:
            kappa_min, kappa_max = kappa_max, kappa_min
        node.setParam("dB/dr", kappa)
        return (kappa_min, kappa_max)

    def set_quads_from_mstate(self, filename=None, parameter=None, verbose=True):
        """Set quad currents or gradients from .mstate file.
        
        Parameters
        ----------
        filename : str
            Path to mstate file.
        parameter : {"gradient", "current"}
            Parameter to set.
        verbose : bool
            Whether to print updates.
        """
        if filename is None:
            return
        setpoints = dict()
        if value_type == "gradient":
            setpoints = quad_readbacks_from_mstate(filename)
            for quad_name, gradient in setpoints.items():
                self.set_quad_gradient(quad_name, gradient, verbose=verbose)
        elif value_type == "current":
            setpoints = quad_setpoints_from_mstate(filename)
            for quad_name, current in setpoints.items():
                self.set_quad_current(quad_name, current, verbose=verbose)
            
    def set_quads_from_file(self, filename=None, comment="#", verbose=True):
        """Set quadrupole dB/dr directly from file.
        
        Each line gives "quad_name dB/dr". Lines starting with `comment` are skipped.
        """
        if verbose:
            print("Reading file {}".format(filename))
        file = open(filename)
        for i, line in enumerate(file):
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(comment):
                continue
            name, value = line.split(" ")
            name = self.shorten_quad_name(name)
            node = self.magnets[name]["node"]
            value = float(value)
            node.setParam("dB/dr", value)
            if verbose:
                print("Set {} dB/dr={}".format(node.getName(), value))
                
    def set_pmq_gradient(self, quad_name, gradient, verbose=True):
        quad_name = self.shorten_quad_name(quad_name)
        if quad_name not in self.quad_names_fodo_short:
            raise ValueError("Unknown FODO quadrupole '{}'.".format(quad_name))
        node = self.magnets[quad_name]["node"]
        length = node.getLength()
        kappa = -gradient / length
        self.magnets[name]["node"].setParam("dB/dr", kappa)
        if verbose:
            print("Updated {} dB/dr={:.3f} (GL={:.3f}, L={:.3f})".format(quad_name, kappa, gradient, length))
            
    def set_pmq_length(self, quad_name, length, verbose=True):
        quad_name = self.shorten_quad_name(quad_name)
        if quad_name not in self.quad_names_fodo_short:
            raise ValueError("Unknown FODO quadrupole '{}'.".format(quad_name))
        node = self.magnets[quad_name]["node"]
        gradient = -node.getParam("dB/dr") * node.getLength()
        node.setLength(length)
        kappa = -gradient / node.getLength()
        node.setParam("dB/dr", kappa)
        if verbose:
            print("Updated {} dB/dr={:.3f} (GL={:.3f}, L={:.3f})".format(quad_name, kappa, gradient, length))

    def set_overlapping_pmq_fields(self, z_step=0.001, verbose=True):
        if verbose:
            for name in self.quad_names_fodo:
                print("Replacing {} with overlapping field model.".format(name))
        sequences = self.lattice.getSequences()
        sequence_names = [sequence.getName() for sequence in sequences[-1:]]
        Replace_Quads_to_OverlappingQuads_Nodes(
            self.lattice,
            z_step,
            accSeq_Names=sequence_names,
            quad_Names=self.quad_names_fodo,
            EngeFunctionFactory=get_quad_func,
        )
        # Update FODO quad names.
        for i, name in enumerate(self.quad_names_fodo):
            self.quad_names_fodo[i] = "{}:group:1:OVRLPQ".format(name)
        
    def set_linac_tracker(self, setting):
        """Use linac-style quads and drifts instead of TEAPOT-style. 
        
        This is useful when the energy spread is large, but is slower and is not symplectic.
        """
        self.lattice.setLinacTracker(setting)
        
    def set_fringe_fields(self, setting):
        for node in self.lattice.getNodes():
            try:
                node.setUsageFringeFieldIN(setting)
                node.setUsageFringeFieldOUT(setting)
            except:
                pass

    def add_space_charge_nodes(
        self,
        grid_size_x=64,
        grid_size_y=64,
        grid_size_z=64,
        path_length_min=0.01,
        n_bunches=3,
        verbose=True,
    ):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
        sc_calc = SpaceChargeCalc3D(grid_size_x, grid_size_y, grid_size_z)
        if n_bunches > 1:
            sc_calc.numExtBunches(n_bunches)
            sc_calc.freqOfBunches(self.rf_frequency)
        sc_nodes = setSC3DAccNodes(self.lattice, path_length_min, sc_calc)
        if _mpi_rank == 0 and verbose:
            print(
                "Added {} space charge nodes".format(len(sc_nodes)),
                "(grid={}X{}X{}, n_bunches={}, path_length_min={})".format(
                    grid_size_x,
                    grid_size_y,
                    grid_size_z,
                    n_bunches,
                    path_length_min,
                ),
            )
        self.space_charge_nodes = sc_nodes
        return self.space_charge_nodes
    
    def add_uniform_ellipsoid_space_charge_nodes(
        self,
        n_ellipsoids=5,
        path_length_min=0.010,
        verbose=True,
    ):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
        sc_calc = SpaceChargeCalcUnifEllipse(n_ellipsoids)
        sc_nodes = setUniformEllipsesSCAccNodes(self.lattice, path_length_min, sc_calc)
        if _mpi_rank == 0 and verbose:
            print("Added {} uniform ellipsoid space charge nodes".format(len(sc_nodes)))
        self.space_charge_nodes = sc_nodes
        return self.space_charge_nodes

    def add_aperture_nodes(self, drift_step=0.1, start=0.0, stop=None, verbose=True):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
        aperture_nodes = Add_quad_apertures_to_lattice(self.lattice)
        aperture_nodes = Add_bend_apertures_to_lattice(self.lattice, aperture_nodes, step=drift_step)
        if stop is None:
            stop = self.lattice.getLength()
        aperture_nodes = Add_drift_apertures_to_lattice(
            self.lattice,
            start,
            stop,
            drift_step,
            self.pipe_diameter,
            aperture_nodes,
        )
        if _mpi_rank == 0 and verbose:
            print("Added {} aperture nodes.".format(len(aperture_nodes)))
            print(
                "pipe_diameter={}, drift_step={}, start={}, stop={}".format(
                    self.pipe_diameter, drift_step, start, stop,
                )
            )
        self.aperture_nodes = aperture_nodes
        return self.aperture_nodes
    
    def save_node_positions(self, filename="lattice_nodes.txt"):
        file = open(filename, "w")
        file.write("node position length\n")
        for node in self.lattice.getNodes():
            file.write("{} {} {}\n".format(node.getName(), node.getPosition(), node.getLength()))
        file.close()
        
    def save_lattice_structure(self, filename="lattice_structure.txt"):
        file = open(filename, "w")
        file.write(self.lattice.structureToText())
        file.close()
        
    def save_quad_strengths(self, filename="quad_settings.dat"):
        file = open(filename, "w")
        file.write("# quad_name dB/dr\n")
        for node in self.lattice.getQuads():
            file.write("{} {}\n".format(node.getName(), node.getParam("dB/dr")))
        file.close()
                