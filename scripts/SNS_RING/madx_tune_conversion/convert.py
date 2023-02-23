"""
The lattice tunes in PyORBIT may be slightly different than those input to the
MADX file. This program adjusts the input tunes to MADX until the output tunes
from PyORBIT are correct.

To run: `./START.sh convert.py 1`
"""
import fileinput
import subprocess
import sys

import numpy as np
from scipy.optimize import least_squares

from bunch import Bunch
from orbit.teapot import teapot
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.utils.consts import mass_proton

# from cpymad.madx import Madx
# madx = Madx()

def set_madx_file_tunes(script_name, nux_madx=1.0, nuy_madx=1.0):
    """Replace tunes in madx script."""
    for line in fileinput.input([script_name], inplace=True):
        text = line.strip()
        if text.startswith('QH:='):
            line = 'QH:={};\n'.format(nux_madx)
        elif text.startswith('QV:='):
            line = 'QV:={};\n'.format(nuy_madx)
        sys.stdout.write(line)
        
        
def set_input_file(script_name, latfile):
    prefix = "CALL, file = "
    new_line = "".join([prefix, "'{}';\n".format(latfile)])
    for line in fileinput.input([script_name], inplace=True):
        if line.strip().startswith(prefix):
            line = new_line
        sys.stdout.write(line)
    
        
def set_output_file(script_name, latfile):
    """Set the output lattice file name in madx script."""
    prefix = 'SAVE,sequence=RNGINJ, FILE='
    new_line = ''.join([prefix, "'{}',clear;\n".format(latfile)])
    for line in fileinput.input([script_name], inplace=True):
        if line.strip().startswith(prefix):
            line = new_line
        sys.stdout.write(line)
        
        
def run_madx(script_name, hide_output=True, python=False):
    """Run madx script."""
    if python:
        madx.option(echo=(not hide_output))
        madx.call(file=script_name)
    else:
        cmd = './madx {} > /dev/null 2>&1' if hide_output else './madx {}'
        subprocess.call(cmd.format(script_name), shell=True)
            
            
def get_tunes(file, seq, fringe=False, mass=mass_proton, kin_energy=1.0):
    """Create PyORBIT lattice from file and calculate the tunes."""
    lattice = TEAPOT_Lattice()
    lattice.readMADX(file, seq)
    for node in lattice.getNodes():
        node.setUsageFringeFieldIN(fringe)
        node.setUsageFringeFieldOUT(fringe)
    _bunch = Bunch()
    _bunch.mass(mass)
    _bunch.getSyncParticle().kinEnergy(kin_energy)
    matrix_lattice = TEAPOT_MATRIX_Lattice(lattice, _bunch)
    ring_params = matrix_lattice.getRingParametersDict()
    nux = ring_params['fractional tune x']
    nuy = ring_params['fractional tune y']
    return nux, nuy
           
           
def madx_to_pyorbit_tunes(
    madx_script, nux, nuy, latfile, latseq, atol, rtol=1e-4, seed=None, fringe=False,
):
    """Find correct MADX tunes for desired PyORBIT tunes.
    
    The input tunes to madx are iteratively corrected until the correct
    value in PyORBIT is reached.
    
    Parameters
    ----------
    madx_script : str
        Name of madx script which generates lattice.
    nux{nuy} : float
        The desired horizontal{vertical} tune.
    latfile : str
        The name of the lattice file output by the MADX script.
    latseq : str
        The name of the `sequence` keyword in the MADX script.
    atol : float
        Tolerance for absolute difference b/w MADX and PyORBIT tunes.
    rtol : float
        Tolerance for absolute relative difference between input tunes on
        subsequent iterations.
    fringe : bool
        Whether to use fringe field calculations.
        
    Returns
    -------
    nux_madx, nuy_madx: The horizontal and vertical tunes to input to madx.
    error_x, error_y: The differences between target and actual tunes.
    """
    set_output_file(madx_script, latfile)
    
    nux_madx, nuy_madx = nux, nuy
    converged_x = converged_y = False
    max_iters = 1000
    for _ in range(max_iters):
        set_madx_file_tunes(madx_script, nux_madx, nuy_madx)
        run_madx(madx_script)
        nux_calc, nuy_calc = get_tunes(latfile, latseq, fringe)
        print 'MADX tunes:    {}, {}'.format(nux_madx, nuy_madx)
        print 'PyORBIT tunes: {}, {}'.format(nux_calc, nuy_calc)
        print ''
        error_x = (nux % 1) - nux_calc
        error_y = (nuy % 1) - nuy_calc
        converged_x = abs(error_x) < atol or abs(error_x/nux) < rtol
        converged_y = abs(error_y) < atol or abs(error_y/nuy) < rtol
        if not converged_x:
            nux_madx += error_x
        if not converged_y:
            nuy_madx += error_y
        if converged_x and converged_y:
            return nux_madx, nuy_madx, error_x, error_y

        
if __name__ == '__main__':

    # Settings
    nux = 6.180  # desired nux
    nuy = 6.180  # desired nux
    atol = 1.0e-5  # absolute tolerance
    madx_script = 'SNS_RING_chromaticity.mad'
    input_file = "SNS_RING_dual_solenoid.lat"
    latfile = "LATTICE.lat"
    latseq = 'rnginjsol'
    fringe = False

    # Remove old output
    subprocess.call('rm ./_output/*', shell=True)
    
    set_input_file(madx_script, input_file)

    # Find correct madx inputs
    nux_madx, nuy_madx, error_x, error_y = madx_to_pyorbit_tunes(
        madx_script, nux, nuy, latfile, latseq, atol, fringe=fringe,
    )
    print('Done.')
    print 'error_x = {:.5f}'.format(error_x)
    print 'error_y = {:.5f}'.format(error_y)

    # Save lattice file and correct inputs
    file = open('./_output/madx_to_pyorbit.dat', 'w')
    file.write('MADX tunes to get correct PyORBIT tunes: \n')
    file.write('    nux_madx = {}\n'.format(nux_madx))
    file.write('    nuy_madx = {}\n'.format(nuy_madx))
    file.close()
    subprocess.call('mv LATTICE.lat ./_output', shell=True)
    subprocess.call('mv madx.ps optics optics_for_G4BL twiss ./_output', shell=True)
