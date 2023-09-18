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


class MADXScript:
    def __init__(
        self, 
        filename=None,
        input_filename="sns_ring.lat",
        output_lattice_filename="sns_ring.lattice", 
        output_lattice_seq="rnginjsol",
        madx_path=None,
    ):
        self.filename = filename
        self.input_filename = input_filename
        self.output_lattice_filename = output_lattice_filename
        self.output_lattice_seq = output_lattice_seq
        self.madx_path = madx_path
        
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
            cmd = "{} {}".format(self.madx_path, self.filename)
            if not verbose:
                cmd = "{} > /dev/null 2>&1".format(cmd)  
            subprocess.call(cmd, shell=True)
            
    def move_output(self, outdir):
        filenames = [
            self.output_lattice_filename,
            "ac_twiss",
            "ringinj_twiss",
            "sp_second_match_twiss",
            "sp_twiss",
            "twiss",
            "twiss_noSol",
            "twiss_ripken",
        ]
        for filename in filenames:
            try:
                subprocess.call("mv {} {}".format(filename, outdir), shell=True)
            except:
                pass
            
        
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
            self.madx_script.set_tunes(nux_madx, nuy_madx)
            self.madx_script.run(verbose=0)
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
 