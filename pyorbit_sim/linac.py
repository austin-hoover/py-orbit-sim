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

import pyorbit_sim.bunch_utils


class BunchWriter:
    def __init__(self, folder="./", prefix=None, index=0, position=0.0, verbose=True):
        self.folder = folder
        self.prefix = prefix
        self.index = index
        self.position = position
        self.verbose = verbose
        
    def action(self, bunch, node=None, position=None):  
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

        filename = "bunch"
        if self.prefix is not None:
            filename = "{}_{}".format(self.prefix, filename)
        if self.index is not None:
            filename = "{}_{}".format(filename, self.index)
        if node is not None:
            filename = "{}_{}".format(filename, node)
        filename = "{}.dat".format(filename)
        filename = os.path.join(self.folder, filename)
        if _mpi_rank == 0 and self.verbose:
            print("Writing bunch to file {}".format(filename))
        bunch.dumpBunch(filename)
        self.index += 1
        self.position = position
        
        
class BunchWriterNode(BaseLinacNode):
    def __init__(self, name="bunch_writer_node", node=None, writer=None, **kws):
        BaseLinacNode.__init__(self, name)
        self.writer = writer
        if self.writer is None:
            self.writer = BunchWriter(**kws)
        self.active = True
        self.node = node
        
    def track(self, params_dict):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
        if self.active and params_dict.has_key("bunch"):
            self.writer.action(params_dict["bunch"], node=self.node)
            
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
        track_history=True,
        track_rms=True,
        dispersion_flag=False,
        emit_norm_flag=False,
        position_offset=0.0,
        verbose=True,
        rf_frequency=402.5e6,
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
                "write_bunch": call `bunch.dumpBunch`
                "plot_bunch": call plotting routines
        track_history : bool
            Whether to append to history array on each action.
        track_rms : bool
            Whether include RMS bunch parameters in history arrays.
        emit_norm_flag, dispersion_flag : bool
            Used by `BunchTwissAnalysis` class.
        position_offset : float
            The initial position in the lattice [m].
        verbose : bool
            Whether to print an update statement on each action.
        """
        self.position = self.position_offset = position_offset
        self.step = 0
        self.start_time = None
        self.rf_frequency = rf_frequency

        self.stride = stride
        if self.stride is None:
            self.stride = dict()
        self.stride.setdefault("update", 0.1)
        self.stride.setdefault("write_bunch", np.inf)
        self.stride.setdefault("plot_bunch", np.inf)
        self.writer = writer
        self.plotter = plotter
        
        self.dispersion_flag = int(dispersion_flag)
        self.emit_norm_flag = int(emit_norm_flag)
        self.track_history = track_history
        self.track_rms = track_rms
        self.verbose = verbose
        
        self.history = dict()
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
        ]
        for i in range(6):
            keys.append("mean_{}".format(i))
        for i in range(6):
            for j in range(i + 1):
                keys.append("cov_{}-{}".format(j, i))
        for key in keys:
            self.history[key] = []
                        
    def action(self, params_dict):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
        
        # Update position; decide whether to proceed.        
        position = params_dict["path_length"] + self.position_offset
        if (position - self.position) < self.stride["update"]:
            return
        
        self.position = position
        
        bunch = params_dict["bunch"]
        node = params_dict["node"]
        beta = bunch.getSyncParticle().beta()
        gamma = bunch.getSyncParticle().gamma()
        n_parts = bunch.getSizeGlobal()
        if self.start_time is None:
            self.start_time = time.clock()
        time_ellapsed = time.clock() - self.start_time
        
        # Record scalar values (position, energy, etc.)
        if _mpi_rank == 0 and self.track_history:
            self.history["position"].append(position)
            self.history["node"].append(node.getName())
            self.history["n_parts"].append(n_parts)
            self.history["gamma"].append(gamma)
            self.history["beta"].append(beta)
            self.history["energy"].append(bunch.getSyncParticle().kinEnergy())

        # Record covariance matrix.
        if self.track_history and self.track_rms:
            bunch_twiss_analysis = BunchTwissAnalysis()
            order = 2
            bunch_twiss_analysis.computeBunchMoments(bunch, order, self.dispersion_flag, self.emit_norm_flag)
            for i in range(6):
                key = "mean_{}".format(i)
                value = bunch_twiss_analysis.getAverage(i)
                if _mpi_rank == 0:
                    self.history[key].append(value)
            for i in range(6):
                for j in range(i + 1):
                    key = "cov_{}-{}".format(j, i)
                    value = bunch_twiss_analysis.getCorrelation(j, i)
                    if _mpi_rank == 0:
                        self.history[key].append(value)
                               
        if _mpi_rank == 0 and self.track_rms:
            x_rms = np.sqrt(self.history["cov_0-0"][-1])
            y_rms = np.sqrt(self.history["cov_2-2"][-1])
            z_rms = np.sqrt(self.history["cov_4-4"][-1])
            z_to_phase_coeff = pyorbit_sim.bunch_utils.get_z_to_phase_coeff(bunch, self.rf_frequency)
            z_rms_deg = -z_to_phase_coeff * z_rms
            self.history["x_rms"].append(x_rms)
            self.history["y_rms"].append(y_rms)
            self.history["z_rms"].append(z_rms)
            self.history["z_rms_deg"].append(z_rms_deg)
            self.history["z_to_phase_coeff"].append(z_to_phase_coeff)
                        
        # Print update statement.
        if self.verbose and _mpi_rank == 0:
            if self.track_rms:
                fstr = "{:>5} | {:>10.2f} | {:>7.3f} | {:>8.4f} | {:>9.3f} | {:>9.3f} | {:>10.3f} | {:<9.3e} | {} "
                if self.step == 0:
                    print(
                        "{:<5} | {:<10} | {:<7} | {:<8} | {:<5} | {:<9} | {:<10} | {:<9} | {}"
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
                fstr = "{:>5} | {:>10.2f} | {:>7.3f} | {:>8.4f} | {:<9.3e} | {} "
                if self.step == 0:
                    print(
                        "{:<5} | {:<10} | {:<7} | {:<10} | {:<9} | {}"
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
        if self.writer is not None and self.stride["write_bunch"] is not None:
            if (position - self.writer.position) >= self.stride["write_bunch"]:
                self.writer.action(bunch, node=node.getName(), position=position)

        # Call plotting routines.
        if self.plotter is not None and self.stride["plot_bunch"] is not None:
            if (position - self.plotter.position) >= self.stride["plot_bunch"]:
                info = dict()
                for key in self.history:
                    if self.history[key]:
                        info[key] = self.history[key][-1]
                info["node"] = node.getName()
                info["step"] = self.step
                info["position"] = position
                info["gamma"] = gamma
                info["beta"] = beta
                self.plotter.action(bunch, info=info, verbose=self.verbose)  # MPI?

    def clear_history(self):
        """Clear history array."""
        for key in self.history:
            self.history[key] = []

    def write_history(self, filename=None, delimiter=","):
        """Write history array to file."""
        if not self.track_history:
            print("Nothing to write! self.track_history=False")
            return
        keys = list(self.history)
        data = np.array([self.history[key] for key in keys]).T
        df = pd.DataFrame(data=data, columns=keys)
        df.to_csv(filename, sep=delimiter, index=False)
        return df


def track(bunch, lattice, monitor=None, start=0.0, stop=None, verbose=True):
    """Track bunch from start to stop."""
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

    # Get start/stop node names, indices, and positions.
    if stop is None:
        stop = lattice.getLength()
    start = _get_node(start, lattice)
    stop = _get_node(stop, lattice)
    
    # Add monitor.
    action_container = AccActionsContainer("monitor")
    if monitor is not None:
        monitor.position_offset = start["s0"]
        action_container.addAction(monitor.action, AccActionsContainer.EXIT)
        
    params_dict = dict()
    params_dict["lostbunch"] = Bunch()

    if _mpi_rank == 0 and verbose:
        print("Tracking from {} to {}.".format(start["node"].getName(), stop["node"].getName()))

    time_start = time.clock()
    lattice.trackBunch(
        bunch,
        paramsDict=params_dict,
        actionContainer=action_container,
        index_start=start["index"],
        index_stop=stop["index"],
    )
    
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


def _get_node(argument, lattice):
    """Helper method for `track_bunch` and `track_bunch_reverse`."""
    if type(argument) is str:
        name = argument
        node = lattice.getNodeForName(name)
        index = lattice.getNodeIndex(node)
        s0 = node.getPosition() - 0.5 * node.getLength()
        s1 = node.getPosition() + 0.5 * node.getLength()
    else:
        position = argument
        node, index, s0, s1 = lattice.getNodeForPosition(position)
    return {
        "node": node,
        "index": index,
        "s0": s0,
        "s1": s1,
    }