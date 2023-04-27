"""SNS Linac simulation.

The lattice can be modified by replacing the BaseRF_Gap nodes with AxisFieldRF_Gap
nodes for the selected sequences. These nodes will use the RF fields at the axis
of the RF gap to track the bunch. The usual BaseRF_Gap nodes have a zero length. 

Apertures are added to the lattice.
"""
from __future__ import print_function
import math
import random
import sys
import time

from bunch import Bunch, BunchTwissAnalysis
from linac import BaseRfGap
from linac import BaseRfGap_slow
from linac import MatrixRfGap
from linac import RfGapThreePointTTF_slow
from linac import RfGapTTF
from linac import RfGapTTF_slow
from orbit.bunch_generators import GaussDist3D
from orbit.bunch_generators import KVDist3D
from orbit.bunch_generators import TwissContainer
from orbit.bunch_generators import WaterBagDist3D
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
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
from orbit.space_charge.sc2p5d import setSC2p5DrbAccNodes
from spacecharge import SpaceChargeCalc3D
from spacecharge import SpaceChargeCalcUnifEllipse
from sns_linac_bunch_generator import SNS_Linac_BunchGenerator


random.seed(100)

names = [
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

# Make lattice from XML file.
max_drift_length = 0.01  # [m]
xml_file_name = "./scripts/SNS_LINAC/data/sns_linac.xml"
sns_linac_factory = SNS_LinacLatticeFactory()
sns_linac_factory.setMaxDriftLength(max_drift_length)
lattice = sns_linac_factory.getLinacAccLattice(names, xml_file_name)
print("Linac lattice is ready. L= {}".format(lattice.getLength()))

# Set up RF gap model.
# There are three available models at this moment:
##     BaseRfGap uses only E0TL*cos(phi)*J0(kr) with E0TL = const.
##     MatrixRfGap uses a matrix approach like envelope codes.
##     RfGapTTF uses Transit Time Factors (TTF) like PARMILA.
cppGapModel = RfGapTTF  # {BaseRfGap, BaseRfGap_slow, MatrixRfGap, MatrixRfGap_slow, RfGapTTF_slow}
for rf_gap in lattice.getRF_Gaps():
    rf_gap.setCppGapModel(cppGapModel())


# --------------------------------------------------------------------------------------
# BaseRF_Gap to  AxisFieldRF_Gap direct replacement.
#
# This could be done directly in {MEBT, CCL, SCLMed, SCLHigh} because RF fields cover
# drifts only. The DTL needs special treatment.
# --------------------------------------------------------------------------------------

dir_location = "./scripts/SNS_LINAC/data/fields.xml"
z_step = 0.002  # longitudinal step along the distributed fields lattice

## Only RF gaps will be replaced with non-zero length models. Quads stay hard-edged. 
## Such approach will not work for DTL cavities - RF and quad fields are overlapped 
## for DTL.
# Replace_BaseRF_Gap_to_AxisField_Nodes(
#     lattice, z_step, dir_location, ["MEBT", "CCL1", "CCL2", "CCL3", "CCL4", "SCLMed"]
# )

## Hard-edge quad models will be replaced with soft-edge models. It is possible for DTL 
## also - if the RF gap models are zero-length ones. 
# accseq_names = ["MEBT", "DTL1", "DTL2", "DTL3", "DTL4", "DTL5", "DTL6"]
# Replace_Quads_to_OverlappingQuads_Nodes(
#     lattice, z_step, accSeq_names, [], SNS_EngeFunctionFactory
# )

## Hard-edge quad and zero-length RF gap models will be replaced with soft-edge quads
## and field-on-axis RF gap models. It can be used for any sequences, no limitations:
# accseq_names = ["MEBT", "DTL1", "DTL2", "DTL3", "DTL4", "DTL5", "DTL6"]
# Replace_BaseRF_Gap_and_Quads_to_Overlapping_Nodes(
#     lattice, z_step, dir_location, accSeq_names, [], SNS_EngeFunctionFactory
# )

print("Linac lattice has been modified. New L[m] = ", lattice.getLength())


# --------------------------------------------------------------------------------------
# Space charge
# --------------------------------------------------------------------------------------

sc_path_length_min = 0.01
space_charge_solver = "ellipsoid"
if space_charge_solver == "ellipsoid":
    nEllipses = 1
    calcUnifEllips = SpaceChargeCalcUnifEllipse(nEllipses)
    space_charge_nodes = setUniformEllipsesSCAccNodes(
        lattice, sc_path_length_min, calcUnifEllips
    )
elif space_charge_solver == "3D":
    sizeX = 64
    sizeY = 64
    sizeZ = 64
    calc3d = SpaceChargeCalc3D(sizeX,sizeY,sizeZ)
    space_charge_nodes = setSC3DAccNodes(lattice, sc_path_length_min, calc3d)

max_sc_length = 0.0
min_sc_length = lattice.getLength()
for sc_node in space_charge_nodes:
    scL = sc_node.getLengthOfSC()
    if scL > max_sc_length:
        max_sc_length = scL
    if scL < min_sc_length:
        min_sc_length = scL
        
print("Set up space charge nodes.")
print("maximum SC length =", max_sc_length, "  min=", min_sc_length)


# --------------------------------------------------------------------------------------
# Apertures
# --------------------------------------------------------------------------------------

print("===== Aperture Nodes START  =======")
aprtNodes = Add_quad_apertures_to_lattice(lattice)
aprtNodes = Add_rfgap_apertures_to_lattice(lattice, aprtNodes)
aprtNodes = AddMEBTChopperPlatesAperturesToSNS_Lattice(lattice, aprtNodes)

x_size = 0.042
y_size = 0.042
aprtNodes = AddScrapersAperturesToLattice(
    lattice, "MEBT_Diag:H_SCRP", x_size, y_size, aprtNodes
)

x_size = 0.042
y_size = 0.042
aprtNodes = AddScrapersAperturesToLattice(
    lattice, "MEBT_Diag:V_SCRP", x_size, y_size, aprtNodes
)
print("===== Aperture Nodes Added =======")
for node in aprtNodes:
    print("aprt=", node.getName(), " pos =", node.getPosition())



# Set linac-style quads and drifts instead of TEAPOT style. (This is useful when 
# the energy spread is large, but is slower and is not symplectic.)
linac_tracker = True
lattice.setLinacTracker(linac_tracker)


# --------------------------------------------------------------------------------------
# Initial bunch
# --------------------------------------------------------------------------------------

# Transverse emittances are unnormalized and in pi*mm*mrad.
# Longitudinal emittance is in pi*eV*sec.
e_kin_ini = 0.0025  # [GeV]
beam_current = 38.0  # [mA]
mass = 0.939294  # [GeV]
gamma = (mass + e_kin_ini) / mass
beta = math.sqrt(gamma * gamma - 1.0) / gamma
print("relat. gamma=", gamma)
print("relat.  beta=", beta)
frequency = 402.5e6
v_light = 2.99792458e8  # in [m/sec]

# Emittances are normalized - transverse by gamma*beta and long. by gamma**3*beta.
(alphaX, betaX, emittX) = (-1.9620, 0.1831, 0.21)
(alphaY, betaY, emittY) = (1.7681, 0.1620, 0.21)
(alphaZ, betaZ, emittZ) = (0.0196, 0.5844, 0.24153)

alphaZ = -alphaZ

# Make emittances un-normalized XAL units [m*rad].
emittX = 1.0e-6 * emittX / (gamma * beta)
emittY = 1.0e-6 * emittY / (gamma * beta)
emittZ = 1.0e-6 * emittZ / (gamma**3 * beta)

print(" ========= XAL Twiss ===========")
print(" aplha beta emitt[mm*mrad] X= %6.4f %6.4f %6.4f " % (alphaX, betaX, emittX * 1.0e6))
print(" aplha beta emitt[mm*mrad] Y= %6.4f %6.4f %6.4f " % (alphaY, betaY, emittY * 1.0e6))
print(" aplha beta emitt[mm*mrad] Z= %6.4f %6.4f %6.4f " % (alphaZ, betaZ, emittZ * 1.0e6))

# ---- long. size in mm
sizeZ = math.sqrt(emittZ * betaZ) * 1.0e3

# ---- transform to pyORBIT emittance[GeV*m]
emittZ = emittZ * gamma**3 * beta**2 * mass
betaZ = betaZ / (gamma**3 * beta**2 * mass)

print(" ========= PyORBIT Twiss ===========")
print(" aplha beta emitt[mm*mrad] X= %6.4f %6.4f %6.4f " % (alphaX, betaX, emittX * 1.0e6))
print(" aplha beta emitt[mm*mrad] Y= %6.4f %6.4f %6.4f " % (alphaY, betaY, emittY * 1.0e6))
print( " aplha beta emitt[mm*MeV] Z= %6.4f %6.4f %6.4f " % (alphaZ, betaZ, emittZ * 1.0e6))

twissX = TwissContainer(alphaX, betaX, emittX)
twissY = TwissContainer(alphaY, betaY, emittY)
twissZ = TwissContainer(alphaZ, betaZ, emittZ)

print("Start Bunch Generation.")
bunch_gen = SNS_Linac_BunchGenerator(twissX, twissY, twissZ)
bunch_gen.setKinEnergy(e_kin_ini)
bunch_gen.setBeamCurrent(beam_current)
bunch_in = bunch_gen.getBunch(nParticles=int(1e5), distributorClass=WaterBagDist3D)
print("Bunch Generation completed.")


# --------------------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------------------

lattice.trackDesignBunch(bunch_in)
print("Design tracking completed.")

pos_start = 0.0
params_dict = {"old_pos": -1.0, "count": 0, "pos_step": 0.1}
actionContainer = AccActionsContainer("Test Design Bunch Tracking")
twiss_analysis = BunchTwissAnalysis()

def action_entrance(params_dict):
    node = params_dict["node"]
    bunch = params_dict["bunch"]
    pos = params_dict["path_length"]
    if params_dict["old_pos"] == pos:
        return
    if params_dict["old_pos"] + params_dict["pos_step"] > pos:
        return
    params_dict["old_pos"] = pos
    params_dict["count"] += 1
    gamma = bunch.getSyncParticle().gamma()
    beta = bunch.getSyncParticle().beta()
    twiss_analysis.analyzeBunch(bunch)
    x_rms = 1000.0 * math.sqrt(twiss_analysis.getTwiss(0)[1] * twiss_analysis.getTwiss(0)[3])
    y_rms = 1000.0 * math.sqrt(twiss_analysis.getTwiss(1)[1] * twiss_analysis.getTwiss(1)[3])
    z_rms = 1000.0 * math.sqrt(twiss_analysis.getTwiss(2)[1] * twiss_analysis.getTwiss(2)[3])
    z_to_phase_coeff = bunch_gen.getZtoPhaseCoeff(bunch)
    z_rms_deg = z_to_phase_coeff * z_rms / 1000.0
    nParts = bunch.getSizeGlobal()
    (alphaX, betaX, emittX) = (
        twiss_analysis.getTwiss(0)[0],
        twiss_analysis.getTwiss(0)[1],
        twiss_analysis.getTwiss(0)[3] * 1.0e6,
    )
    (alphaY, betaY, emittY) = (
        twiss_analysis.getTwiss(1)[0],
        twiss_analysis.getTwiss(1)[1],
        twiss_analysis.getTwiss(1)[3] * 1.0e6,
    )
    (alphaZ, betaZ, emittZ) = (
        twiss_analysis.getTwiss(2)[0],
        twiss_analysis.getTwiss(2)[1],
        twiss_analysis.getTwiss(2)[3] * 1.0e6,
    )
    norm_emittX = emittX * gamma * beta
    norm_emittY = emittY * gamma * beta
    # ---- phi_de_emittZ will be in [pi*deg*MeV]
    phi_de_emittZ = z_to_phase_coeff * emittZ
    eKin = bunch.getSyncParticle().kinEnergy() * 1.0e3
    s = " %35s  %4.5f " % (node.getName(), pos + pos_start)
    s += "   %6.4f  %6.4f  %6.4f  %6.4f   " % (alphaX, betaX, emittX, norm_emittX)
    s += "   %6.4f  %6.4f  %6.4f  %6.4f   " % (alphaY, betaY, emittY, norm_emittY)
    s += "   %6.4f  %6.4f  %6.4f  %6.4f   " % (alphaZ, betaZ, emittZ, phi_de_emittZ)
    s += "   %5.3f  %5.3f  %5.3f " % (x_rms, y_rms, z_rms_deg)
    s += "  %10.6f   %8d " % (eKin, nParts)
    file_out.write(s + "\n")
    file_out.flush()
    s_prt = " %5d  %35s  %4.5f " % (
        params_dict["count"],
        node.getName(),
        pos + pos_start,
    )
    s_prt += "  %5.3f  %5.3f   %5.3f " % (x_rms, y_rms, z_rms_deg)
    s_prt += "  %10.6f   %8d " % (eKin, nParts)
    print(s_prt)


def action_exit(params_dict):
    action_entrance(params_dict)


actionContainer.addAction(action_entrance, AccActionsContainer.ENTRANCE)
actionContainer.addAction(action_exit, AccActionsContainer.EXIT)

file_out = open("pyorbit_twiss_sizes_ekin.dat", "w")
s = " Node   position "
s += "   alphaX betaX emittX  normEmittX"
s += "   alphaY betaY emittY  normEmittY"
s += "   alphaZ betaZ emittZ  emittZphiMeV"
s += "   sizeX sizeY sizeZ_deg"
s += "   eKin Nparts "
file_out.write(s + "\n")

print(" N node   position    sizeX  sizeY  sizeZdeg  eKin Nparts ")

time_start = time.clock()

lattice.trackBunch(bunch_in, paramsDict=params_dict, actionContainer=actionContainer)

print("time [sec] = {}".format(time.clock() - time_start))
file_out.close()