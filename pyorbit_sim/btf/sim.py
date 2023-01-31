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

from ..bunch_utils import reverse_bunch


class Monitor:
    """Monitor the bunch.

    The `action` method in this class records various beam parameters; it may
    also take photographs of the beam.
    
    Should perhaps make this more general.
    
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
    """

    def __init__(
        self,
        start_position=0.0,
        plotter=None,
        dispersion_flag=False,
        emit_norm_flag=False,
        pos_step=0.005,
    ):
        self.start_position = start_position
        self.plotter = plotter
        self.dispersion_flag = dispersion_flag
        self.emit_norm_flag = emit_norm_flag
        self.pos_step = 0.005
        keys = [
            "position",
            "node",
            "n_parts",
            "n_lost",
            "gamma",
            "beta",
        ]
        for i in range(6):
            keys.append("mean_{}".format(i))
        for i in range(6):
            for j in range(i + 1):
                keys.append("cov_{}-{}".format(j, i))
        self.history = {key: [] for key in keys}

    def action(self, params_dict):
        node = params_dict["node"]
        bunch = params_dict["bunch"]
        position = params_dict["path_length"] + self.start_position
        if params_dict["old_pos"] == position:
            return
        if params_dict["old_pos"] + self.pos_step > position:
            return
        params_dict["old_pos"] = position
        params_dict["count"] += 1
 
        # Print update statement.
        n_steps = params_dict["count"]
        n_parts = bunch.getSizeGlobal()
        print(
            "step={}, s={:.3f} [m], node={}, n_parts={}"
            .format(n_steps, position, node.getName(), n_parts)
        )

        # Update history.
        self.history["position"].append(position)
        self.history["node"].append(node.getName())
        self.history["n_parts"].append(n_parts)
        self.history["n_lost"].append(self.history["n_parts"][0] - n_parts)
        self.history["gamma"].append(bunch.getSyncParticle().gamma())
        self.history["beta"].append(bunch.getSyncParticle().beta())
        bunch_twiss_analysis = BunchTwissAnalysis()
        order = 2
        bunch_twiss_analysis.computeBunchMoments(
            bunch, 
            order, 
            self.dispersion_flag, 
            self.emit_norm_flag
        )
        for i in range(6):
            key = "mean_{}".format(i)
            self.history[key].append(bunch_twiss_analysis.getAverage(i))
        for i in range(6):
            for j in range(i + 1):
                key = "cov_{}-{}".format(j, i)
                self.history[key].append(bunch_twiss_analysis.getCorrelation(j, i))            

        # Make plots.
        if self.plotter is not None:
            info = dict()
            for key in self.history:
                if self.history[key]:
                    info[key] = self.history[key][-1]
            info["node"] = params_dict["node"].getName()
            info["step"] = params_dict["count"]
            self.plotter.plot(get_bunch_coords(bunch), info=info, verbose=True)

    def forget(self):
        for key in self.history:
            self.history[key] = []

    def write(self, filename=None, delimiter=","):
        keys = list(self.history)
        data = np.array([self.history[key] for key in keys]).T
        df = pd.DataFrame(data=data, columns=keys)
        df.to_csv(filename, sep=delimiter, index=False)
        return df


def _get_node(argument, lattice):
    """Return (node, node_index, position_start, position_stop)."""
    if type(argument) is str:
        node = lattice.getNodeForName(argument)
        return (
            node,
            lattice.getNodeIndex(node),
            node.getPosition() - 0.5 * node.getLength(),
            node.getPosition() + 0.5 * node.getLength(),
        )
    else:
        return lattice.getNodeForPosition(position)


def track_bunch(bunch, lattice, monitor=None, start=0.0, stop=None, verbose=0):
    """Track bunch from start to stop."""
    if stop is None:
        stop = lattice.getLength()     
    node_start, index_start, s0_start, s1_start = _get_node(start, lattice)
    node_stop, index_stop, s0_stop, s1_stop = _get_node(stop, lattice)
        
    action_container = AccActionsContainer("monitor")
    if monitor is not None:
        monitor.start_position = s0_start
        action_container.addAction(monitor.action, AccActionsContainer.EXIT)

    if verbose:
        print("Tracking from {} to {}.".format(start, stop))
        
    time_start = time.clock()
        
    params_dict = dict()
    params_dict["old_pos"] = -1.0
    params_dict["count"] = 0
        
    lattice.trackBunch(
        bunch,
        paramsDict=params_dict,
        actionContainer=action_container,
        index_start=index_start,
        index_stop=index_stop,
    )

    # Save the last time step.
    if monitor is not None:
        monitor.action(params_dict)
    params_dict["old_pos"] = -1
    
    if verbose:
        print("time = {:.3f} [sec]".format(time.clock() - time_start))


def track_bunch_reverse(bunch, lattice, monitor=None, start=0.0, stop=None, verbose=0):
    """Untested!"""
    if type(start) is float:
        start = self.lattice.getLength() - start
    if type(stop) is float:
        stop = self.lattice.getLength() - stop
    lattice.reverseOrder()
    bunch = reverse_bunch(bunch)
    track_bunch(bunch, lattice, monitor=monitor, start=stop, stop=start, verbose=verbosef)
    lattice.reverseOrder()
    bunch = reverse_bunch(bunch)
    if monitor is not None:
        node_stop, index_stop, s0_stop, s1_stop = _get_node(stop, lattice)
        monitor.history["position"] = 2.0 * s1_stop - monitor.history["position"]