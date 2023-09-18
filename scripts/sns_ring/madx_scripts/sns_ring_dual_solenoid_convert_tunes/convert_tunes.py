from __future__ import print_function
import argparse
import fileinput
import os
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import least_squares

from bunch import Bunch
from orbit.teapot import teapot
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.utils.consts import mass_proton

sys.path.insert(0, "/home/46h/repo/py-orbit-sim/scripts/sns_ring/")
from tune_converter import MADXScript
from tune_converter import TuneConverter


MADX_PATH = "/home/46h/repo/madx/madx-linux64-intel"
    
    
parser = argparse.ArgumentParser()
parser.add_argument("--nux", type=float, default=6.175)
parser.add_argument("--nuy", type=float, default=6.175)
parser.add_argument("--rtol", type=float, default=1.00e-05)
parser.add_argument("--atol", type=float, default=1.00e-05)
parser.add_argument("--madx-path", type=str, default=MADX_PATH)
parser.add_argument("--madx-script", type=str, default="sns_ring.mad")
parser.add_argument("--madx-input", type=str, default="sns_ring.lat")
parser.add_argument("--madx-output-lattice", type=str, default="sns_ring.lattice")
parser.add_argument("--madx-output-seq", type=str, default="rnginjsol")
parser.add_argument("--fringe", type=int, default=0)
parser.add_argument("--max-iters", type=int, default=1000)
args = parser.parse_args()

timestamp = time.strftime("%y%m%d%H%M%S")
outdir = "./data_output/{}/".format(timestamp)
if not os.path.exists(outdir):
    os.makedirs(outdir)

madx_script = MADXScript(
    filename=args.madx_script,
    madx_path=args.madx_path,
)
madx_script.set_input_filename(args.madx_input)
madx_script.set_output_lattice_filename(args.madx_output_lattice)

tune_converter = TuneConverter(
    madx_script=madx_script, 
    fringe=args.fringe,
    mass=mass_proton, 
    kin_energy=1.0,
)

nux_madx, nuy_madx, nux_pyorbit, nuy_pyorbit = tune_converter.convert_tunes(
    nux=args.nux, 
    nuy=args.nuy, 
    max_iters=args.max_iters,
    rtol=args.rtol,
    atol=args.atol,
)
madx_script.move_output(outdir)

# # # Save lattice file and correct inputs
# # file = open(os.path.join(outdir, "info.dat"), "w")
# # file.write("MADX tunes to get correct PyORBIT tunes: \n")
# # file.write("nux_madx = {}\n".format(nux_madx))
# # file.write("nuy_madx = {}\n".format(nuy_madx))
# # file.write("nux_pyorbit = {}\n".format(nux_pyorbit))
# # file.write("nuy_pyorbit = {}\n".format(nuy_pyorbit))
# # file.close()


# sys.exit()