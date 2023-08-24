"""Print min/max quadrupole strengths."""
from __future__ import print_function
import os
from pprint import pprint

from orbit.py_linac.lattice import Quad

from sns_btf import SNS_BTF


file_dir = os.path.dirname(os.path.realpath(__file__))
input_dir = os.path.join(file_dir, "data_input")
xml_filename = os.path.join(input_dir, "xml/btf_lattice_straight.xml")
coef_filename = os.path.join(input_dir, "magnets/default_i2gl_coeff_straight.csv")

linac = SNS_BTF(coef_filename=coef_filename, rf_frequency=402.5e06)
lattice = linac.init_lattice(
    xml_filename=xml_filename,
    sequences=["MEBT1", "MEBT2"],
    max_drift_length=0.010,
)
for quad_name in linac.quad_names_no_fodo:
    curr_min, curr_max = linac.get_quad_current_limits(quad_name)
    kappa_min, kappa_max = linac.get_quad_kappa_limits(quad_name)
    print(
        "{}: Imin={:.3f} Imax={:.3f} kmin={:.3f} kmax={:.3f}".format(
            quad_name, curr_min, curr_max, kappa_min, kappa_max
        )
    )