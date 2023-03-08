"""Helpers for linac simulations."""
from __future__ import print_function
import os
import sys
import time

import numpy as np
import pandas as pd

from bunch import Bunch
from bunch import BunchTwissAnalysis
from orbit.diagnostics.diagnostics import get_bunch_coords
from orbit.lattice import AccActionsContainer
import orbit.utils.consts as consts
import orbit_mpi

from pyorbit_sim.bunch_utils import reverse_bunch


class Monitor:
    """Monitor the bunch.

    The `action` method in this class records various beam parameters; it may
    also take photographs of the beam.

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
    start_position : float
        The start position in the lattice [m].
    plotter : btfsim.plot.Plotter
        Plotting manager. We can decide when this plotter should activate.
    emit_norm_flag, dispersion_flag : bool
        Used by `BunchTwissAnalysis` class.
    track_history : bool
        Whether to append to history array on each action.
    track_rms : bool
        Whether include RMS bunch parameters in history arrays.
    verbose : bool
        Whether to print an update statement on each action.
    """

    def __init__(
        self,
        start_position=0.0,
        plotter=None,
        dispersion_flag=False,
        emit_norm_flag=False,
        track_history=True,
        track_rms=True,
        verbose=True,
    ):
        self.start_position = start_position
        self.plotter = plotter
        self.dispersion_flag = int(dispersion_flag)
        self.emit_norm_flag = int(emit_norm_flag)
        self.verbose = verbose
        self.start_time = None
        self.track_history = track_history
        self.track_rms = track_rms
        self.verbose = verbose
        self.history = dict()
        keys = [
            "position",
            "node",
            "gamma",
            "beta",
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

        bunch = params_dict["bunch"]
        node = params_dict["node"]
        step = params_dict["step"]
        position = params_dict["path_length"] + self.start_position
        beta = bunch.getSyncParticle().beta()
        gamma = bunch.getSyncParticle().gamma()
        if params_dict["old_pos"] == position:
            return
        if params_dict["old_pos"] + 0.0001 > position:
            return
        if self.start_time is None:
            self.start_time = time.clock()
        time_ellapsed = time.clock() - self.start_time
        if self.verbose and _mpi_rank == 0:
            print(
                "step={}, time={:.3f}, s={:.3f}, node={}".format(
                    step, time_ellapsed, position, node.getName()
                )
            )
        params_dict["step"] += 1
        if _mpi_rank == 0 and self.track_history:
            self.history["position"].append(position)
            self.history["node"].append(node.getName())
            self.history["beta"].append(beta)
            self.history["gamma"].append(gamma)

        if self.track_history and self.track_rms:
            bunch_twiss_analysis = BunchTwissAnalysis()
            order = 2
            bunch_twiss_analysis.computeBunchMoments(
                bunch, order, self.dispersion_flag, self.emit_norm_flag
            )
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

        if self.plotter is not None:
            # This section is untested.
            info = dict()
            for key in self.history:
                if self.history[key]:
                    info[key] = self.history[key][-1]
            info["node"] = params_dict["node"].getName()
            info["step"] = params_dict["step"]
            # Will need to update get_bunch_coords to get global bunch coordinates.
            self.plotter.plot(get_bunch_coords(bunch), info=info, verbose=True)

    def forget(self):
        for key in self.history:
            self.history[key] = []

    def write(self, filename=None, delimiter=","):
        if not self.track_history:
            print("Nothing to write! self.track_history=False")
            return
        keys = list(self.history)
        data = np.array([self.history[key] for key in keys]).T
        df = pd.DataFrame(data=data, columns=keys)
        df.to_csv(filename, sep=delimiter, index=False)
        return df


def _get_node(argument, lattice):
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


def track_bunch(bunch, lattice, monitor=None, start=0.0, stop=None, verbose=True):
    """Track bunch from start to stop."""
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)

    if stop is None:
        stop = lattice.getLength()
    start = _get_node(start, lattice)
    stop = _get_node(stop, lattice)

    action_container = AccActionsContainer("monitor")
    if monitor is not None:
        monitor.start_position = start["s0"]
        action_container.addAction(monitor.action, AccActionsContainer.EXIT)

    params_dict = dict()
    params_dict["old_pos"] = -1.0
    params_dict["step"] = 0

    if _mpi_rank == 0 and verbose:
        print(
            "Tracking from {} to {}.".format(
                start["node"].getName(), stop["node"].getName()
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

    if verbose and _mpi_rank == 0:
        print("time = {:.3f} [sec]".format(time.clock() - time_start))


def track_bunch_reverse(bunch, lattice, monitor=None, start=None, stop=0.0, verbose=0):
    """Track bunch backward from stop to start."""
    lattice.reverseOrder()
    bunch = reverse_bunch(bunch)
    track_bunch(bunch, lattice, monitor=monitor, start=stop, stop=start, verbose=verbose)
    lattice.reverseOrder()
    bunch = reverse_bunch(bunch)
    if monitor is not None:
        # node_stop, index_stop, s0_stop, s1_stop = _get_node(stop, lattice)
        # monitor.history["position"] = 2.0 * s1_stop - monitor.history["position"]
        pass
