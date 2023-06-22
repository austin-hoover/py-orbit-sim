"""SNS linac simulation."""
from __future__ import print_function
import os
import sys

from bunch import Bunch
from linac import BaseRfGap
from linac import BaseRfGap_slow
from linac import MatrixRfGap
from linac import RfGapThreePointTTF
from linac import RfGapThreePointTTF_slow
from linac import RfGapTTF
from linac import RfGapTTF_slow
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
from orbit.py_linac.lattice_modifications import Add_quad_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_rfgap_apertures_to_lattice
from orbit.py_linac.lattice_modifications import AddMEBTChopperPlatesAperturesToSNS_Lattice
from orbit.py_linac.lattice_modifications import AddScrapersAperturesToLattice
from orbit.py_linac.lattice_modifications import GetLostDistributionArr
from orbit.py_linac.lattice_modifications import Replace_BaseRF_Gap_and_Quads_to_Overlapping_Nodes
from orbit.py_linac.lattice_modifications import Replace_BaseRF_Gap_to_AxisField_Nodes
from orbit.py_linac.lattice_modifications import Replace_Quads_to_OverlappingQuads_Nodes
from orbit.py_linac.linac_parsers import SNS_LinacLatticeFactory
from orbit.py_linac.overlapping_fields import SNS_EngeFunctionFactory
from orbit.space_charge.sc3d import setSC3DAccNodes
from orbit.space_charge.sc3d import setUniformEllipsesSCAccNodes
from orbit.space_charge.sc2p5d import setSC2p5DrbAccNodes
from orbit.utils import consts
import orbit_mpi
from spacecharge import SpaceChargeCalc3D
from spacecharge import SpaceChargeCalcUnifEllipse


class SNS_LINAC:
    def __init__(self, input_dir=None, xml_filename=None, rf_frequency=402.5e+06):
        self.input_dir = input_dir
        self.xml_filename = os.path.join(self.input_dir, xml_filename)
        self.lattice = None
        self.sequences = None
        self.rf_frequency = rf_frequency
        self.aperture_nodes = []
        
    def initialize(self, sequences=None, max_drift_length=0.010, verbose=True):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

        self.sequences = sequences
        sns_linac_factory = SNS_LinacLatticeFactory()
        sns_linac_factory.setMaxDriftLength(max_drift_length)
        self.lattice = sns_linac_factory.getLinacAccLattice(self.sequences, self.xml_filename)
        if _mpi_rank == 0 and verbose:
            print("Initialized lattice.")
            print("XML filename = {}".format(self.xml_filename))
            print("lattice length = {:.3f} [m])".format(self.lattice.getLength()))
        return self.lattice
    
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
    
    def set_rf_gap_model(self, rf_gap_model):
        for rf_gap in self.lattice.getRF_Gaps():
            rf_gap.setCppGapModel(rf_gap_model)
            
    def set_overlapping_rf_and_quad_fields(self, sequences=None, z_step=0.002, xml_filename="sns_rf_fields.xml"):
        if sequences is None:
            sequences = self.sequences    
        fields_filename = os.path.join(self.input_dir, xml_filename)

        # Replace hard-edge quads with soft-edge quads; replace zero-length RF gap models
        # with field-on-axis RF gap models. Can be used for any sequences, no limitations.
        Replace_BaseRF_Gap_and_Quads_to_Overlapping_Nodes(
            self.lattice, z_step, fields_filename, sequences, [], SNS_EngeFunctionFactory
        )

        # Add tracking through the longitudinal field component of the quad. The
        # longitudinal component is nonzero only for the distributed magnetic field
        # of the quad. 
        for node in self.lattice.getNodes():
            if (isinstance(node, OverlappingQuadsNode) or isinstance(node, AxisField_and_Quad_RF_Gap)):
                node.setUseLongitudinalFieldOfQuad

    def set_linac_tracker(self, setting):
        # Use linac-style quads and drifts instead of TEAPOT style. (Useful when 
        # the energy spread is large, but is slower and is not symplectic.)
        self.lattice.setLinacTracker(True)

    def add_space_charge_nodes(
        self, 
        solver="FFT", 
        grid_size=(64, 64, 64), 
        n_ellipsoids=5, 
        path_length_min=0.010, 
        verbose=True,
    ):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
        
        sc_nodes = None
        if solver == "FFT":
            calc = SpaceChargeCalc3D(*grid_size)
            sc_nodes = setSC3DAccNodes(self.lattice, path_length_min, calc)
        elif solver == "ellipsoid":
            calc = SpaceChargeCalcUnifEllipse(n_ellipsoids)
            sc_nodes = setUniformEllipsesSCAccNodes(self.lattice, path_length_min, calc)
        if (sc_nodes is not None) and (_mpi_rank == 0) and verbose:
            lengths = [node.getLengthOfSC() for node in sc_nodes]
            min_length = min(min(lengths), self.lattice.getLength())
            max_length = max(max(lengths), 0.0)
            print("Added {} space charge nodes (solver={})".format(len(sc_nodes), solver))
            print("min length = {}".format(min_length))
            print("max length = {}".format(max_length))
        return sc_nodes
    
    def add_transverse_aperture_nodes(self, x_size=0.042, y_size=0.042, verbose=True):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

        aperture_nodes = Add_quad_apertures_to_lattice(self.lattice)
        aperture_nodes = Add_rfgap_apertures_to_lattice(self.lattice, aperture_nodes)
        aperture_nodes = AddMEBTChopperPlatesAperturesToSNS_Lattice(self.lattice, aperture_nodes)
        aperture_nodes = AddScrapersAperturesToLattice(self.lattice, "MEBT_Diag:H_SCRP", x_size, y_size, aperture_nodes)
        aperture_nodes = AddScrapersAperturesToLattice(self.lattice, "MEBT_Diag:V_SCRP", x_size, y_size, aperture_nodes)
        n_transverse_aperture_nodes = len(aperture_nodes)
        if _mpi_rank == 0 and verbose:
            print("Added {} transverse aperture nodes.".format(len(aperture_nodes)))
        self.aperture_nodes.extend(aperture_nodes)
    
    def add_longitudinal_apertures(
        self, 
        classes=None, 
        phase_min=-180.0, 
        phase_max=+180.0, 
        energy_min=-0.100, 
        energy_max=+0.100, 
        verbose=True
    ):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

        if classes is None:
            classes = [
                BaseRF_Gap, 
                AxisFieldRF_Gap, 
                AxisField_and_Quad_RF_Gap,
                Quad, 
                OverlappingQuadsNode,
            ]
        node_pos_dict = self.lattice.getNodePositionsDict()
        aperture_nodes = []
        for node in self.lattice.getNodesOfClasses(classes):
            if node.hasParam("aperture") and node.hasParam("aprt_type"):
                position_start, position_stop = node_pos_dict[node]

                aperture_node = LinacPhaseApertureNode(frequency=self.rf_frequency, name="{}_phase_aprt_out".format(node.getName()))
                aperture_node.setMinMaxPhase(phase_min, phase_max)  # [deg]
                aperture_node.setPosition(position_stop)
                aperture_node.setSequence(node.getSequence())
                node.addChildNode(aperture_node, node.EXIT)
                aperture_nodes.append(aperture_node)

                # Energy apertures are probably unnecessary, but add them anyway.
                aperture_node = LinacEnergyApertureNode(name="{}_energy_aprt_out".format(node.getName()))
                aperture_node.setMinMaxEnergy(energy_min, energy_max)  # [GeV]
                aperture_node.setPosition(position_stop)
                aperture_node.setSequence(node.getSequence())
                node.addChildNode(aperture_node, node.EXIT)
                aperture_nodes.append(aperture_node)
        if _mpi_rank == 0 and verbose:
            print("Added {} longitudinal aperture nodes.".format(len(aperture_nodes)))
        self.aperture_nodes.extend(aperture_nodes)