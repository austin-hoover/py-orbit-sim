"""SNS linac lattice generation."""
from __future__ import print_function
import collections
import os
import sys

from linac import BaseRfGap
from linac import BaseRfGap_slow
from linac import MatrixRfGap
from linac import RfGapThreePointTTF_slow
from linac import RfGapTTF
from linac import RfGapTTF_slow
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.py_linac.lattice.LinacApertureNodes import LinacApertureNode
from orbit.py_linac.lattice_modifications import Add_bend_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_drift_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_quad_apertures_to_lattice
from orbit.py_linac.lattice_modifications import Add_rfgap_apertures_to_lattice
from orbit.py_linac.lattice_modifications import AddMEBTChopperPlatesAperturesToSNS_Lattice
from orbit.py_linac.lattice_modifications import AddScrapersAperturesToLattice
from orbit.py_linac.lattice_modifications import Replace_BaseRF_Gap_and_Quads_to_Overlapping_Nodes
from orbit.py_linac.lattice_modifications import Replace_BaseRF_Gap_to_AxisField_Nodes
from orbit.py_linac.lattice_modifications import Replace_Quads_to_OverlappingQuads_Nodes
from orbit.py_linac.linac_parsers import SNS_LinacLatticeFactory
from orbit.py_linac.overlapping_fields import SNS_EngeFunctionFactory
from orbit.space_charge.sc3d import setSC3DAccNodes
from orbit.space_charge.sc3d import setUniformEllipsesSCAccNodes
from orbit.utils.xml import XmlDataAdaptor
import orbit_mpi
from spacecharge import SpaceChargeCalc3D
from spacecharge import SpaceChargeCalcUnifEllipse

from bunch_generator import SNS_Linac_BunchGenerator

    
class SNS_LINAC:
    """Class to generate SNS linac lattice."""

    def __init__(self, xml="/data/sns_linac.xml", max_drift_length=0.01, sequences=None, verbose=True):
        """Constructor.

        Parameters
        ----------
        xml
        max_drift_length
        sequences
        """
        self.sequences = sequences
        if self.sequences is None:
            self.sequences = [
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
        sns_linac_factory = SNS_LinacLatticeFactory()
        sns_linac_factory.setMaxDriftLength(max_drift_length)
        self.lattice = sns_linac_factory.getLinacAccLattice(self.sequences, xml)
        if verbose:
            print("Loaded lattice from file.")
            print("XML filename = {}".format(xml))
            print("lattice length = {:.3f} [m])".format(self.lattice.getLength()))
        
    def set_rf_gap_model(self, model=None):
        """Set the RF gap model.

        Parameters
        ----------
        model : class
            * BaseRfGap / BaseRfGap_slow: Uses only E0TL * cos(phi) * J0(kr), with E0TL = const.
            * MatrixRfGap / MatrixRfGap_slow: uses a matrix approach like envelope codes.
            * RfGapTTF / RfGapTTF_slow: uses Transit Time Factors (TTF) like PARMILA.
        """
        for rf_gap in self.lattice.getRF_Gaps():
            rf_gap.setCppGapModel(model())
            
    def add_space_charge_nodes(
        self,
        grid_size=(64, 64, 64), 
        path_length_min=0.01,
        n_bunches=1,
        freq=402.5e-6,
        solver="3D",
        n_ellipsoids=1,
        verbose=True,
    ):
        """Add space charge nodes to the lattice."""
        if solver == "ellipsoid":
            sc_calc = SpaceChargeCalcUnifEllipse(n_ellipsoids)
            sc_nodes = setUniformEllipsesSCAccNodes(self.lattice, path_length_min, sc_calc)
        else:
            sc_calc = SpaceChargeCalc3D(grid_size[0], grid_size[1], grid_size[2])
            if n_bunches > 1: 
                sc_calc.numExtBunches(n_bunches)
                sc_calc.freqOfBunches(freq)
            sc_nodes = setSC3DAccNodes(self.lattice, path_length_min, sc_calc)
        if verbose:
            print("Added {} space charge nodes".format(len(sc_nodes)))
            if solver == "ellipsoid":
                print(
                    "(grid={}X{}X{}, n_bunches={}, path_length_min={})".format(
                        grid_size[0], 
                        grid_size[1], 
                        grid_size[2], 
                        n_bunches, 
                        path_length_min,
                    )
                )
            else:
                print("n_ellipsoids = {}".format(n_ellipsoids))
            sc_lengths = [sc_node.getLengthOfSC() for sc_node in sc_nodes]
            min_sc_length = min(min(sc_lengths), self.lattice.getLength())
            max_sc_length = max(max(sc_lengths), 0.0)
            print("max length = {}".format(max_sc_length))
            print("min length = {}".format(min_sc_length))
        return sc_nodes

    def add_aperture_nodes(self, x_size=0.042, y_size=0.042, verbose=True):
        aperture_nodes = Add_quad_apertures_to_lattice(self.lattice)
        aperture_nodes = Add_rfgap_apertures_to_lattice(self.lattice, aperture_nodes)
        aperture_nodes = AddMEBTChopperPlatesAperturesToSNS_Lattice(self.lattice, aperture_nodes)
        aperture_nodes = AddScrapersAperturesToLattice(self.lattice, "MEBT_Diag:H_SCRP", x_size, y_size, aperture_nodes)
        aperture_nodes = AddScrapersAperturesToLattice(self.lattice, "MEBT_Diag:V_SCRP", x_size, y_size, aperture_nodes)
        if verbose:
            print("Added {} aperture nodes.".format(len(aperture_nodes)))
        return aperture_nodes
    
    def replace_base_rf_gap_to_axis_field_nodes(
        self, 
        z_step=0.002, 
        fields_filename="", 
        seq_names=["MEBT", "CCL1", "CCL2", "CCL3", "CCL4", "SCLMed"],
    ):
        """"Calls `Replace_BaseRF_Gap_to_AxisField_Nodes`.
        
        Only RF gaps will be replaced with non-zero length models. Quads stay hard-edged. 
        Such approach will not work for DTL cavities - RF and quad fields are overlapped 
        for DTL.

        Parameters
        ----------
        z_step : float
            Longitudinal step size along the distributed fields lattice.
        fields_filename : str
            XML file containing
        seq_names : list[str]
            The sequence names.
        """
        Replace_BaseRF_Gap_to_AxisField_Nodes(self.lattice, z_step, fields_filename, seq_names)
        
    def replace_quads_to_overlapping_quads_nodes(
        self, 
        z_step=0.002, 
        seq_names=["MEBT", "DTL1", "DTL2", "DTL3", "DTL4", "DTL5", "DTL6"],
    ):
        """Calls `Replace_Quads_to_OverlappingQuads_Nodes`.
        
        Hard-edge quad models will be replaced with soft-edge models. It is possible for DTL 
        also - if the RF gap models are zero-length ones. 
        
        Parameters
        ----------
        z_step : float
            Longitudinal step size along the distributed fields lattice.
        seq_names : list[str]
            The sequence names.
        """
        Replace_Quads_to_OverlappingQuads_Nodes(self.lattice, z_step, seq_names, [], SNS_EngeFunctionFactory)

    def replace_base_rf_gap_and_quads_to_overlapping_nodes(
        self,
        z_step=0.002,
        fields_filename="",
        seq_names=["MEBT", "DTL1", "DTL2", "DTL3", "DTL4", "DTL5", "DTL6"], 
    ):
        """Calls `Replace_BaseRF_Gap_and_Quads_to_Overlapping_Nodes`.
        
        Hard-edge quad and zero-length RF gap models will be replaced with soft-edge quads
        and field-on-axis RF gap models. Can be used for any sequences, no limitations.
        
        Parameters
        ----------
        z_step : float
            Longitudinal step size along the distributed fields lattice.
        fields_filename : str
            XML file containing
        seq_names : list[str]
            The sequence names.
        """
        Replace_BaseRF_Gap_and_Quads_to_Overlapping_Nodes(
            self.lattice, z_step, fields_filename, seq_names, [], SNS_EngeFunctionFactory
        )