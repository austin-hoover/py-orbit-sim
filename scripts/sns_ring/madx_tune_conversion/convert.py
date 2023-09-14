"""Convert MADX tunes to PyORBIT tunes.

As of 2023-09-14, the PyORBIT and MADX give different tunes. This script 
adjusts the input tunes to MADX until the output tunes from PyORBIT are
correct.

Run this script from this directory: 

`${ORBIT_ROOT}/bin/pyORBIT convert.py --nux=6.18 --nuy=6.18`
"""
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

# from cpymad.madx import Madx
# madx = Madx()


class MadxScript:
    def __init__(
        self, 
        filename=None,
        input_filename="sns_ring.lat",
        output_lattice_filename="sns_ring.lattice", 
        output_lattice_seq="rnginjsol",
        outdir=".",
    ):
        self.filename = filename
        self.input_filename = input_filename
        self.output_lattice_filename = output_lattice_filename
        self.output_lattice_seq = output_lattice_seq
        self.outdir = outdir
        
    def replace_line(self, old_line, new_line):
        for line in fileinput.input([self.filename], inplace=True):
            if line.strip().startswith(old_line):
                line = new_line
            sys.stdout.write(line)
        
    def set_input_filename(self, filename):
        self.input_filename = filename
        old_line = "CALL, file = "
        new_line = "".join([old_line, "'{}';\n".format(filename)])
        self.replace_line(old_line, new_line)
            
    def set_output_lattice_filename(self, filename=None):
        self.output_lattice_filename = filename
        old_line = "SAVE,sequence={}, FILE=".format(self.output_lattice_seq.upper())
        new_line = "".join([old_line, "'{}',clear;\n".format(filename)])
        self.replace_line(old_line, new_line)
     
    def set_tunes(self, nux=6.18, nuy=6.18):
        self.replace_line("QH:=", "QH:={:0.8f};\n".format(nux))
        self.replace_line("QV:=", "QV:={:0.8f};\n".format(nuy))
            
    def run(self, python=False, verbose=0):
        if python:
            madx.option(echo=(not hide_output))
            madx.call(file=self.filename)
        else:
            cmd = "./madx {}" if verbose else "./madx {} > /dev/null 2>&1"
            cmd = cmd.format(self.filename)
            subprocess.call(cmd, shell=True)
            
    def move_output(self, outdir):
        subprocess.call("mv {} {}".format(self.output_lattice_filename, outdir), shell=True)
        subprocess.call("mv madx.ps optics optics_for_G4BL twiss {}".format(outdir), shell=True)

        
class TuneConverter:
    def __init__(self, madx_script=None, fringe=False, mass=mass_proton, kin_energy=1.0):
        self.madx_script = madx_script
        self.fringe = False
        self.mass = mass
        self.kin_energy = kin_energy
      
    def get_pyorbit_tunes(self):
        """Create PyORBIT lattice from file and calculate the tunes."""
        lattice = TEAPOT_Lattice()
        lattice.readMADX(
            self.madx_script.output_lattice_filename, 
            self.madx_script.output_lattice_seq
        )
        for node in lattice.getNodes():
            node.setUsageFringeFieldIN(self.fringe)
            node.setUsageFringeFieldOUT(self.fringe)
        test_bunch = Bunch()
        test_bunch.mass(self.mass)
        test_bunch.getSyncParticle().kinEnergy(self.kin_energy)
        matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, test_bunch)
        ring_params = matrix_lattice.getRingParametersDict()
        nux = ring_params["fractional tune x"]
        nuy = ring_params["fractional tune y"]
        return nux, nuy
    
    def convert_tunes(
        self,
        nux=6.180,
        nuy=6.180,
        max_iters=1000,
        rtol=1.00e-04,
        atol=1.00e-04,
        lb=6.01,
        ub=6.32,
    ):
        nux_target = nux_madx = nux
        nuy_target = nuy_madx = nuy
        converged_x = converged_y = False
        for _ in range(max_iters):
            madx_script.set_tunes(nux_madx, nuy_madx)
            madx_script.run(verbose=0)
            nux_pyorbit, nuy_pyorbit = self.get_pyorbit_tunes()
            
            print()
            print(
                "MADX tunes:    {:0.4f}, {:0.4f} (abs {:0.4f}, {:0.4f})".format(
                    nux_madx % 1, nuy_madx % 1, nux_madx, nuy_madx
                )
            )
            print("PyORBIT tunes: {:0.4f}, {:0.4f}".format(nux_pyorbit, nuy_pyorbit))
            print()
            
            error_x = (nux_target % 1) - nux_pyorbit
            error_y = (nuy_target % 1) - nuy_pyorbit
            converged_x = (abs(error_x) < atol) or (abs(error_x / nux) < rtol)
            converged_y = (abs(error_y) < atol) or (abs(error_y / nuy) < rtol)
            if not converged_x:
                nux_madx += error_x
            if not converged_y:
                nuy_madx += error_y
            if converged_x and converged_y:
                return (nux_madx, nuy_madx, nux_pyorbit, nuy_pyorbit)
    

# Settings
parser = argparse.ArgumentParser()
parser.add_argument("--nux", type=float, default=6.180)
parser.add_argument("--nuy", type=float, default=6.180)
parser.add_argument("--rtol", type=float, default=1.00e-05)
parser.add_argument("--atol", type=float, default=1.00e-05)
parser.add_argument("--madx-script", type=str, default="sns_ring.mad")
parser.add_argument("--madx-input", type=str, default="sns_ring_dual_solenoid.lat")
parser.add_argument("--madx-output-lattice", type=str, default="LATTICE.lat")
parser.add_argument("--madx-output-seq", type=str, default="rnginjsol")
parser.add_argument("--fringe", type=int, default=0)
parser.add_argument("--max-iters", type=int, default=1000)
args = parser.parse_args()

# Create output directory.
timestamp = time.strftime("%y%m%d%H%M%S")
outdir = "./data_output/{}/".format(timestamp)
if not os.path.exists(outdir):
    os.makedirs(outdir)
    
madx_script = MadxScript(args.madx_script)
madx_script.set_input_filename(args.madx_input)
madx_script.set_output_lattice_filename(args.madx_output_lattice)

tune_converter = TuneConverter(
    madx_script=madx_script, 
    fringe=args.fringe,
    mass=mass_proton, 
    kin_energy=1.0,
)
(nux_madx, nuy_madx, nux_pyorbit, nuy_pyorbit) = tune_converter.convert_tunes(
    nux=args.nux, 
    nuy=args.nuy, 
    max_iters=args.max_iters,
    rtol=args.rtol,
    atol=args.atol,
)
madx_script.move_output(outdir)

# Save lattice file and correct inputs
file = open(os.path.join(outdir, "info.dat"), "w")
file.write("MADX tunes to get correct PyORBIT tunes: \n")
file.write("nux_madx = {}\n".format(nux_madx))
file.write("nuy_madx = {}\n".format(nuy_madx))
file.write("nux_pyorbit = {}\n".format(nux_pyorbit))
file.write("nuy_pyorbit = {}\n".format(nuy_pyorbit))
file.close()