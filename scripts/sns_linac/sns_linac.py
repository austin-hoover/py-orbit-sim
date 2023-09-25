"""SNS linear accelerator."""
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
from orbit.py_linac.overlapping_fields import SNS_EngeFunctionFactory
from orbit.space_charge.sc3d import setSC3DAccNodes
from orbit.space_charge.sc3d import setUniformEllipsesSCAccNodes
from orbit.space_charge.sc2p5d import setSC2p5DrbAccNodes
from orbit.utils import consts
import orbit_mpi
from spacecharge import SpaceChargeCalc3D
from spacecharge import SpaceChargeCalcUnifEllipse


class SNS_LINAC:
    def __init__(self, rf_frequency=402.5e+06):
        self.lattice = None
        self.rf_frequency = rf_frequency
        self.aperture_nodes = []
        self.sc_nodes = []
        self.sequences = None
        self._sequences = [
            "MEBT",
            "DTL1",
            "DTL2",
            "DTL3",
            "DTL4",
            "DTL5",
            "DTL6",
            "CCL1",
            "CCL2",
            "CCL3",
            "CCL4",
            "SCLMed",
            "SCLHigh",
            "HEBT1",
            "HEBT2",
        ]
                        
    def initialize(
        self, 
        xml_filename=None,
        sequence_start=None, 
        sequence_stop=None, 
        max_drift_length=0.010, 
        verbose=True
    ):    
        lo = self._sequences.index(sequence_start)
        hi = self._sequences.index(sequence_stop)
        self.sequences = self._sequences[lo : hi + 1]
        
        sns_linac_factory = SNS_LinacLatticeFactory()
        sns_linac_factory.setMaxDriftLength(max_drift_length)
        self.lattice = sns_linac_factory.getLinacAccLattice(self.sequences, xml_filename)
        
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
        if _mpi_rank == 0 and verbose:
            print("Initialized lattice.")
            print("XML filename = {}".format(xml_filename))
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
        """Set the RF gap model.
        
        Parameters
        ----------
        rf_gap_model : class
            - {BaseRfGap, BaseRfGap_slow} uses only E0TL * cos(phi) * J0(kr) with E0TL = const.
            - {MatrixRfGap,MatrixRfGap_slow} uses a matrix approach like envelope codes.
            - {RfGapTTF, RfGapTTF_slow} uses Transit Time Factors (TTF) like PARMILA.
            
            The slow variants update all RF gap parameters individually for each particle in the bunch.
        """
        for rf_gap in self.lattice.getRF_Gaps():
            rf_gap.setCppGapModel(rf_gap_model())
            
    def set_overlapping_rf_and_quad_fields(
        self, sequences=None, z_step=0.002, fields_filename="sns_rf_fields.xml"
    ):
        if sequences is None:
            sequences = self.sequences    

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
        """Use linac-style quads and drifts instead of TEAPOT-style. 
        
        This is useful when the energy spread is large, but is slower and is not symplectic.
        """
        self.lattice.setLinacTracker(setting)

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
        sc_nodes = []
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
        self.sc_nodes = sc_nodes
    
    def add_transverse_aperture_nodes(self, scrape_x=0.042, scrape_y=0.042, verbose=True):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
        aperture_nodes = Add_quad_apertures_to_lattice(self.lattice)
        aperture_nodes = Add_rfgap_apertures_to_lattice(self.lattice, aperture_nodes)
        aperture_nodes = AddMEBTChopperPlatesAperturesToSNS_Lattice(self.lattice, aperture_nodes)
        aperture_nodes = AddScrapersAperturesToLattice(
            self.lattice, "MEBT_Diag:H_SCRP", scrape_x, scrape_y, aperture_nodes
        )
        aperture_nodes = AddScrapersAperturesToLattice(
            self.lattice, "MEBT_Diag:V_SCRP", scrape_x, scrape_y, aperture_nodes
        )
        self.aperture_nodes.extend(aperture_nodes)
        if _mpi_rank == 0 and verbose:
            print("Added {} transverse aperture nodes.".format(len(aperture_nodes)))
            
    def add_transverse_aperture_nodes_drifts(
        self, start=0.0, stop=None, step=1.0, radius=0.042, verbose=True,
    ):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
        if stop is None:
            stop = self.lattice.getLength()
        diameter = 2.0 * radius
        aperture_nodes = Add_drift_apertures_to_lattice(self.lattice, start, stop, step, diameter)
        self.aperture_nodes.extend(aperture_nodes)
        if _mpi_rank == 0 and verbose:
            print("Added {} transverse aperture nodes (drift spaces).".format(len(aperture_nodes)))
        return aperture_nodes
            
    def add_phase_aperture_nodes(
        self,
        phase_min=-180.0,  # [GeV]
        phase_max=+180.0,  # [GeV]
        classes=None,
        nametag="phase_aprt",
        verbose=True,
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

        def func(**kws):
            phase_min = kws["phase_min"]
            phase_max = kws["phase_max"]
            aperture_node = LinacPhaseApertureNode(frequency=self.rf_frequency)
            aperture_node.setMinMaxPhase(phase_min, phase_max)  # [deg]
            return aperture_node
                
        aperture_nodes = self._add_aperture_nodes_classes(
            classes=classes,
            nametag=nametag,
            func=func,
            func_kws={
                "phase_min": phase_min,
                "phase_max": phase_max,
            }
        )
        self.aperture_nodes.extend(aperture_nodes)
        if _mpi_rank == 0 and verbose:
            print("Added {} phase aperture nodes.".format(len(aperture_nodes)))
        return aperture_nodes
    
    def add_phase_aperture_nodes_drifts(
        self,
        phase_min=-180.0,  # [deg]
        phase_max=+180.0,  # [deg]
        start=None,
        stop=None,
        step=1.0,
        verbose=True,
    ):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

        def func(**kws):
            phase_min = kws["phase_min"]
            phase_max = kws["phase_max"]
            aperture_node = LinacPhaseApertureNode(frequency=self.rf_frequency)
            aperture_node.setMinMaxPhase(phase_min, phase_max)  # [deg]
            return aperture_node
                
        aperture_nodes = self._add_aperture_nodes_drifts(
            start=start,
            stop=stop,
            step=step,
            nametag="phase_aprt",
            func=func,
            func_kws={
                "phase_min": phase_min,
                "phase_max": phase_max,
            }
        )
        self.aperture_nodes.extend(aperture_nodes)
        if _mpi_rank == 0 and verbose:
            print("Added {} phase aperture nodes (drift spaces).".format(len(aperture_nodes)))
        return aperture_nodes
            
    def add_energy_aperture_nodes(
        self,
        energy_min=-0.100,  # [GeV]
        energy_max=+0.100,  # [GeV]
        classes=None,
        nametag="energy_aprt",
        verbose=True,
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

        def func(**kws):
            energy_min = kws["energy_min"]
            energy_max = kws["energy_max"]
            aperture_node = LinacEnergyApertureNode()
            aperture_node.setMinMaxEnergy(energy_min, energy_max)  # [GeV]
            return aperture_node
                
        aperture_nodes = self._add_aperture_nodes_classes(
            classes=classes,
            nametag=nametag,
            func=func,
            func_kws={
                "energy_min": energy_min,
                "energy_max": energy_max,
            }
        )
        self.aperture_nodes.extend(aperture_nodes)
        if _mpi_rank == 0 and verbose:
            print("Added {} energy aperture nodes.".format(len(aperture_nodes)))
        return aperture_nodes
                    
    def add_energy_aperture_nodes_drifts(
        self,
        energy_min=-0.100,  # [GeV]
        energy_max=+0.100,  # [GeV]
        start=None,
        stop=None,
        step=1.0,
        verbose=True,
    ):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

        def func(**kws):
            energy_min = kws["energy_min"]
            energy_max = kws["energy_max"]
            aperture_node = LinacEnergyApertureNode()
            aperture_node.setMinMaxEnergy(energy_min, energy_max)  # [GeV]
            return aperture_node
                
        aperture_nodes = self._add_aperture_nodes_drifts(
            start=start,
            stop=stop,
            step=step,
            nametag="energy_aprt",
            func=func,
            func_kws={
                "energy_min": energy_min,
                "energy_max": energy_max,
            }
        )
        self.aperture_nodes.extend(aperture_nodes)
        if _mpi_rank == 0 and verbose:
            print("Added {} energy aperture nodes (drift spaces).".format(len(aperture_nodes)))
        return aperture_nodes
    
    def _add_aperture_nodes_classes(
        self,
        classes=None,
        nametag="aprt",
        func=None,
        func_kws=None,
    ):
        """Add aperture nodes to all nodes of a specified class (or classes).
        
        This function is not meant to be called directly. It does not extend
        `self.aperture_nodes`.
        
        Parameters
        ----------
        classes : list
            Parent node classes.
        nametag : str
            Nodes are named "{parent_node_name}_{nametag}_in" and "{parent_node_name}_{nametag}_out".
        func : callable
            Returns an aperture node.
        func_kws : dict
            Key word arguments for `func`. (`aperture_node = func(**func_kws)`).
            
        Returns
        -------
        list[Node]
            The aperture nodes added to the lattice.
        """
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
                for location, suffix, position in zip([node.ENTRANCE, node.EXIT], ["in", "out"], node_pos_dict[node]):
                    aperture_node = func(**func_kws)
                    aperture_node.setName("{}_{}_{}".format(node.getName(), nametag, suffix))
                    aperture_node.setPosition(position)
                    aperture_node.setSequence(node.getSequence())
                    node.addChildNode(aperture_node, location)
                    aperture_nodes.append(aperture_node)
        return aperture_nodes
            
    def _add_aperture_nodes_drifts(
        self,
        start=0.0,
        stop=None,
        step=1.0,
        nametag="aprt",
        func=None,
        func_kws=None,
    ):
        """Add aperture nodes to drift spaces as child nodes.
        
        This function is not meant to be called directly. It does not extend
        `self.aperture_nodes`.
        
        Parameters
        ----------
        start, stop, stop. : float
            Nodes are added between `start` [m] and `stop` [m] with spacing `step` [m].
        nametag : str
            Nodes are named "{parent_node_name}:{part_index}_{nametag}".
        func : callable
            Returns an aperture node.
        func_kws : dict
            Key word arguments for `func`. (`aperture_node = func(**func_kws)`).
            
        Returns
        -------
        list[Node]
            The aperture nodes added to the lattice.
        """
        if func is None:
            return
        if func_kws is None:
            func_kws = dict()
        if stop is None:
            stop = self.lattice.getLength()
            
        node_pos_dict = self.lattice.getNodePositionsDict()
        parent_nodes = self.lattice.getNodesOfClasses([Drift])
                
        last_position, _ = node_pos_dict[parent_nodes[0]]
        last_position = last_position - 2.0 * step        
        child_nodes = []
        for parent_node in parent_nodes:            
            position, _ = node_pos_dict[parent_node]
            if position > stop:
                break
            for index in range(parent_node.getnParts()):
                if start <= position <= stop:
                    if position >= last_position + step:
                        child_node = func(**func_kws)
                        name = "{}".format(parent_node.getName())
                        if parent_node.getnParts() > 1:
                            name = "{}:{}".format(name, index)
                        child_node.setName("{}_{}".format(name, nametag))
                        child_node.setPosition(position)
                        child_node.setSequence(parent_node.getSequence())
                        parent_node.addChildNode(child_node, parent_node.BODY, index, parent_node.BEFORE)
                        child_nodes.append(child_node)
                        last_position = position
                position += parent_node.getLength(index)   
        return child_nodes