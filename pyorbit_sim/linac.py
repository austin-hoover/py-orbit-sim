"""Helpers for linac simulations."""
from __future__ import print_function
import os
import sys
import time

import numpy as np
import pandas as pd

from bunch import Bunch
from bunch import BunchTwissAnalysis
from bunch_utils_functions import copyCoordsToInitCoordsAttr
from orbit.bunch_utils import ParticleIdNumber
from orbit.lattice import AccActionsContainer
from orbit.py_linac.lattice import BaseLinacNode
from orbit.teapot import DriftTEAPOT
import orbit.utils.consts as consts
import orbit_mpi
from orbit_utils import BunchExtremaCalculator

import pyorbit_sim.bunch_utils


_mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
_mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
_mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)


class BunchWriter:
    def __init__(self, outdir="./", index=0, position=0.0, verbose=True):
        self.outdir = outdir
        self.index = index
        self.position = position
        self.verbose = verbose
        
    def action(self, bunch, node_name=None, position=None):  
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

        filename = "bunch"
        if self.index is not None:
            filename = "{}_{:04.0f}".format(filename, self.index)
        if node_name is not None:
            filename = "{}_{}".format(filename, node_name)
        filename = "{}.dat".format(filename)
        filename = os.path.join(self.outdir, filename)
        if _mpi_rank == 0 and self.verbose:
            print("Writing bunch to file {}".format(filename))
        bunch.dumpBunch(filename)
        if self.index is not None:
            self.index += 1
        if position is not None:
            self.position = position
        
        
class BunchWriterNode(BaseLinacNode):
    def __init__(self, name="bunch_writer_node", node_name=None, writer=None, **kws):
        BaseLinacNode.__init__(self, name)
        self.writer = writer
        if self.writer is None:
            self.writer = BunchWriter(**kws)
        self.active = True
        self.node_name = node_name
        
    def track(self, params_dict):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
        if self.active and params_dict.has_key("bunch"):
            self.writer.action(params_dict["bunch"], node_name=self.node_name)
            
    def trackDesign(self, params_dict):
        pass
            

class Monitor:
    """Monitor the bunch.

    Attributes
    ----------
    history : dict[str : list]
        "position" = syncronous particle position [m]
        "node" = node name
        "n_parts" = current number of macroparticles
        "n_lost" = difference between initial/current number of macroparticles.
        "gamma", "beta" = syncronous particle relativistic factors
        "mean_0", "mean_1", ... = first-order moments
        "cov_0-0", "cov_0-1", ... = second-order moments
    position : float
        Last known synchronous particle position.
    writer : BunchWriter
        Manages calls to `bunch.dumpBunch`.
    plotter : Plotter
        Manages plotting routines.
    """
    def __init__(
        self,
        plotter=None,
        writer=None,
        stride=None,
        track_rms=True,
        dispersion_flag=False,
        emit_norm_flag=False,
        position_offset=0.0,
        verbose=True,
        rf_frequency=402.5e6,
        filename=None,
    ):
        """
        Parameters
        ----------
        plotter : Plotter
            Plotting manager.
        writer : Plotter
            Bunch writing manager. 
        stride : dict
            Each value corresponds to the stride length (in meters) between updates.
            If a list of node names is provided, an update will occur only at thos
            nodes.
            The dictionary can contain the following keys.
                "update": proceed with all updates (print statement, etc.)
                "write": call `bunch.dumpBunch`
                "plot": call plotting routines
        track_rms : bool
            Whether include RMS bunch parameters in history arrays.
        emit_norm_flag, dispersion_flag : bool
            Used by `BunchTwissAnalysis` class.
        position_offset : float
            The initial position in the lattice [m].
        verbose : bool
            Whether to print an update statement on each action.
        filename : str or None
            If provided, save the history to a file.
        """
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

        self.position = self.position_offset = position_offset
        self.step = 0
        self.start_time = None
        self.rf_frequency = rf_frequency
        self.dispersion_flag = int(dispersion_flag)
        self.emit_norm_flag = int(emit_norm_flag)
        self.track_rms = track_rms
        self.verbose = verbose

        self.stride = stride
        if self.stride is None:
            self.stride = dict()
        self.stride.setdefault("update", 0.1)
        self.stride.setdefault("write", np.inf)
        self.stride.setdefault("plot", np.inf)
        
        self.writer = writer
        self.plotter = plotter
        
        if _mpi_rank == 0:
            keys = [
                "position",
                "node",
                "n_parts",
                "gamma",
                "beta",
                "energy",
                "x_rms",
                "y_rms",
                "z_rms",
                "z_rms_deg",
                "z_to_phase_coeff",
                "x_min",
                "x_max",
                "y_min",
                "y_max",
                "z_min",
                "z_max",
            ]
            for i in range(6):
                keys.append("mean_{}".format(i))
            for i in range(6):
                for j in range(i + 1):
                    keys.append("cov_{}-{}".format(j, i))
                    
            self.history = dict()
            for key in keys:
                self.history[key] = None

            self.filename = filename
            self.file = None
            if self.filename is not None:
                self.file = open(self.filename, "w")
                line = ",".join(list(self.history))
                line = line[:-1] + "\n"
                self.file.write(line)
            
    def action(self, params_dict, force_update=False):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
        
        # Update position; decide whether to proceed. 
        position = params_dict["path_length"] + self.position_offset
        if not force_update:
            if self.step > 0:
                if (position - self.position) < self.stride["update"]:
                    return
        self.position = position       
        
        # Update clock.
        if self.start_time is None:
            self.start_time = time.clock()
        time_ellapsed = time.clock() - self.start_time
        
        # Get bunch and node.
        bunch = params_dict["bunch"]
        node = params_dict["node"]

        # Record scalar values (position, energy, etc.)
        beta = bunch.getSyncParticle().beta()
        gamma = bunch.getSyncParticle().gamma()
        n_parts = bunch.getSizeGlobal()
        if _mpi_rank == 0:
            self.history["position"] = position
            self.history["node"] = node.getName()
            self.history["n_parts"] = n_parts
            self.history["gamma"] = gamma
            self.history["beta"] = beta
            self.history["energy"] = bunch.getSyncParticle().kinEnergy()

        # Record covariance matrix.
        if self.track_rms:
            twiss_analysis = BunchTwissAnalysis()
            order = 2
            twiss_analysis.computeBunchMoments(bunch, order, self.dispersion_flag, self.emit_norm_flag)
            for i in range(6):
                key = "mean_{}".format(i)
                value = twiss_analysis.getAverage(i)
                if _mpi_rank == 0:
                    self.history[key] = value
            for i in range(6):
                for j in range(i + 1):
                    key = "cov_{}-{}".format(j, i)
                    value = twiss_analysis.getCorrelation(j, i)
                    if _mpi_rank == 0:
                        self.history[key] = value
                                                   
        # Update history array with standard deviations.
        if _mpi_rank == 0 and self.track_rms:
            x_rms = np.sqrt(self.history["cov_0-0"])
            y_rms = np.sqrt(self.history["cov_2-2"])
            z_rms = np.sqrt(self.history["cov_4-4"])
            z_to_phase_coeff = pyorbit_sim.bunch_utils.get_z_to_phase_coeff(bunch, self.rf_frequency)
            z_rms_deg = -z_to_phase_coeff * z_rms
            self.history["x_rms"] = x_rms
            self.history["y_rms"] = y_rms
            self.history["z_rms"] = z_rms
            self.history["z_rms_deg"] = z_rms_deg
            self.history["z_to_phase_coeff"] = z_to_phase_coeff
            
        # Extrema calculations.
        extrema_calculator = BunchExtremaCalculator()
        (x_min, x_max, y_min, y_max, z_min, z_max) = extrema_calculator.extremaXYZ(bunch)
        if _mpi_rank == 0:
            self.history["x_min"] = x_min
            self.history["x_max"] = x_max
            self.history["y_min"] = y_min
            self.history["y_max"] = y_max
            self.history["z_min"] = z_min
            self.history["z_max"] = z_max
                                
        # Print update statement.
        if self.verbose and _mpi_rank == 0:
            if self.track_rms:
                fstr = "{:>5} | {:>10.2f} | {:>10.5f} | {:>8.4f} | {:>9.3f} | {:>9.3f} | {:>10.3f} | {:<9.0f} | {} "
                if self.step == 0:
                    print(
                        "{:<5} | {:<10} | {:<10} | {:<8} | {:<5} | {:<9} | {:<10} | {:<9} | {}"
                        .format("step", "time [s]", "s [m]", "T [MeV]", "xrms [mm]", "yrms [mm]", "zrms [deg]", "nparts", "node")
                    )
                    print(115 * "-")
                print(
                    fstr.format(
                        self.step,
                        time_ellapsed,  # [s]
                        position,  # [m]
                        1000.0 * bunch.getSyncParticle().kinEnergy(),
                        1000.0 * x_rms,
                        1000.0 * y_rms,
                        z_rms_deg,
                        n_parts,
                        node.getName(),
                    )
                )
            else:
                fstr = "{:>5} | {:>10.2f} | {:>10.3f} | {:>8.4f} | {:<9.0f} | {} "
                if self.step == 0:
                    print(
                        "{:<5} | {:<10} | {:<10} | {:<10} | {:<9} | {}"
                        .format("step", "time [s]", "s [m]", "T [MeV]", "nparts", "node")
                    )
                    print(80 * "-")
                print(
                    fstr.format(
                        self.step,
                        time_ellapsed,  # [s]
                        position,  # [m]
                        1000.0 * bunch.getSyncParticle().kinEnergy(),
                        n_parts,
                        node.getName(),
                    )
                )
        self.step += 1
                                                
        # Write bunch coordinates to file.        
        if self.writer is not None and self.stride["write"] is not None:
            if (position - self.writer.position) >= self.stride["write"]:
                self.writer.action(bunch, node_name=node.getName(), position=position)

        # Call plotting routines.
        if self.plotter is not None and self.stride["plot"] is not None and _mpi_rank == 0:
            if (position - self.plotter.position) >= self.stride["plot"]:
                info = dict()
                for key in self.history:
                    if self.history[key]:
                        info[key] = self.history[key]
                info["node"] = node.getName()
                info["step"] = self.step
                info["position"] = position
                info["gamma"] = gamma
                info["beta"] = beta
                self.plotter.action(bunch, info=info, verbose=self.verbose)
                
        # Write new line to history file.
        if _mpi_rank == 0 and self.file is not None:
            data = [self.history[key] for key in self.history]
            line = ""
            for i in range(len(data)):
                line += "{},".format(data[i])
            line = line[:-1] + "\n"
            self.file.write(line)
    
    
def get_node_info(node_name_or_position, lattice):
    """Return node, node index, start and stop position for node name or center position.
    
    Helper method for `track_bunch` and `track_bunch_reverse`.
    
    Parameters
    ----------
    argument : node name or position.
    lattice : LinacAccLattice
    
    Returns
    -------
    dict
        "node": the node instance
        "index": the node index in the lattice
        "s0": the node start position
        "
    """
    if type(node_name_or_position) is str:
        name = node_name_or_position
        node = lattice.getNodeForName(name)
        index = lattice.getNodeIndex(node)
        s0 = node.getPosition() - 0.5 * node.getLength()
        s1 = node.getPosition() + 0.5 * node.getLength()
    else:
        position = node_name_or_position
        node, index, s0, s1 = lattice.getNodeForPosition(position)
    return {
        "node": node,
        "index": index,
        "s0": s0,
        "s1": s1,
    }
    
    
def check_sync_part_time(bunch, lattice, start=0.0, set_design=False, verbose=True):
    """Check if the synchronous particle time is set to the design value at start."""
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
    _mpi_size = orbit_mpi.MPI_Comm_size(_mpi_comm)

    start = get_node_info(start, lattice)
    sync_time = bunch.getSyncParticle().time()
    sync_time_design = 0.0
    if start["index"] > 0:
        design_bunch = lattice.trackDesignBunch(bunch, index_start=0, index_stop=start["index"])
        sync_time_design = design_bunch.getSyncParticle().time()
    if _mpi_rank == 0 and verbose:
        print("Start index = {}:".format(start["index"]))
        print("    Synchronous particle time (actual) = {}".format(sync_time))
        print("    Synchronous particle time (design) = {}".format(sync_time_design))
    if set_design and abs(sync_time - sync_time_design) > 1.0e-30:
        if _mpi_rank == 0 and verbose:
            print("    Setting to design value.")
        bunch.getSyncParticle().time(sync_time_design)
        if _mpi_rank == 0 and verbose:
            print("bunch.getSyncParticle().time() = {}".format(bunch.getSyncParticle().time()))
            

def track(bunch, lattice, monitor=None, start=0.0, stop=None, verbose=True):
    """Track bunch from start to stop."""
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

    # Get start/stop node names, indices, and positions.
    nodes = lattice.getNodes()
    if stop is None:
        stop = nodes[-1].getName()
    start = get_node_info(start, lattice)
    stop = get_node_info(stop, lattice)    
    
    # Add actions.
    action_container = AccActionsContainer("monitor")
    if monitor is not None:
        monitor.position_offset = start["s0"]
        action_container.addAction(monitor.action, AccActionsContainer.ENTRANCE)
        action_container.addAction(monitor.action, AccActionsContainer.EXIT)
        
    # Create params dict and lost bunch.
    params_dict = dict()
    params_dict["lostbunch"] = Bunch()

    if _mpi_rank == 0 and verbose:
        print(
            "Tracking from {} (s={}) to {} (s={}).".format(
                start["node"].getName(), 
                start["s0"],
                stop["node"].getName(),
                stop["s1"],
            )
        )

    time_start = time.clock()
    lattice.trackBunch(
        bunch,
        paramsDict=params_dict,
        actionContainer=action_container,
        index_start=start["index"],
        index_stop=stop["index"],
    )
    monitor.action(params_dict, force_update=True)
    
    if verbose and _mpi_rank == 0:
        print("time = {:.3f} [sec]".format(time.clock() - time_start))
        
    return params_dict


def track_reverse(bunch, lattice, monitor=None, start=None, stop=0.0, verbose=0):
    """Track bunch backward from stop to start."""
    lattice.reverseOrder()
    bunch = pyorbit_sim.bunch_utils.reverse(bunch)
    params_dict = track(bunch, lattice, monitor=monitor, start=stop, stop=start, verbose=verbose)
    lattice.reverseOrder()
    bunch = pyorbit_sim.bunch_utils.reverse(bunch)
    return params_dict