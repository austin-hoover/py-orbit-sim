"""Simulation plotting routines.

These routines use python2-compatible matplotlib; they can be called within
pyorbit scripts.

TODO
----
Compute histogram on each MPI node, then combine and plot the result.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import colors
from matplotlib import patches
from matplotlib import pyplot as plt


import orbit_mpi
from bunch import Bunch
from orbit.teapot import DriftTEAPOT
from orbit.py_linac.lattice import BaseLinacNode


def get_bunch_coords(bunch):
    """Return bunch coordinate array (no MPI)."""
    X = np.zeros((bunch.getSize(), 6))
    for i in range(X.shape[0]):
        X[i, :] = [
            bunch.x(i),
            bunch.xp(i), 
            bunch.y(i), 
            bunch.yp(i), 
            bunch.z(i), 
            bunch.dE(i)
        ]
    return X


# General matplotlib plotting functions.
# -----------------------------------------------------------------------------

DIMS = ["x", "xp", "y", "yp", "z", "dE"]
UNITS = ["m", "rad", "m", "rad", "m", "GeV"]
DIMS_UNITS = ["{} [{}]".format(d, u) for d, u in zip(DIMS, UNITS)]


def centers_from_edges(edges):
    """Compute bin centers from bin edges."""
    return 0.5 * (edges[:-1] + edges[1:])


def ellipse(c1=1.0, c2=1.0, angle=0.0, center=(0, 0), ax=None, **kws):
    """Plot ellipse with semi-axes `c1`,`c2` tilted `angle`radians below the x axis."""
    kws.setdefault("fill", False)
    kws.setdefault("color", "black")
    width = 2.0 * c1
    height = 2.0 * c2
    return ax.add_patch(
        patches.Ellipse(center, width, height, -np.degrees(angle), **kws)
    )


def rms_ellipse_dims(Sigma, axis=(0, 1)):
    """Return dimensions of projected rms ellipse.
    Parameters
    ----------
    Sigma : ndarray, shape (2n, 2n)
        The phase space covariance matrix.
    axis : 2-tuple
        The axis on which to project the covariance ellipsoid. Example: if the
        axes are {x, xp, y, yp}, and axis=(0, 2), then the four-dimensional
        ellipsoid is projected onto the x-y plane.
    ax : plt.Axes
        The ax on which to plot.
    Returns
    -------
    c1, c2 : float
        The ellipse semi-axis widths.
    angle : float
        The tilt angle below the x axis [radians].
    """
    i, j = axis
    sii, sjj, sij = Sigma[i, i], Sigma[j, j], Sigma[i, j]
    angle = -0.5 * np.arctan2(2.0 * sij, sii - sjj)
    sin, cos = np.sin(angle), np.cos(angle)
    c1 = np.sqrt(abs(sii * cos**2 + sjj * sin**2 - 2.0 * sij * sin * cos))
    c2 = np.sqrt(abs(sii * sin**2 + sjj * cos**2 + 2.0 * sij * sin * cos))
    return c1, c2, angle


def rms_ellipse(Sigma=None, center=None, level=1.0, ax=None, **ellipse_kws):
    if type(level) not in [list, tuple, np.ndarray]:
        level = [level]
    c1, c2, angle = rms_ellipse_dims(Sigma)
    for level in level:
        _c1 = c1 * level
        _c2 = c2 * level
        ellipse(_c1, _c2, angle=angle, center=center, ax=ax, **ellipse_kws)
    return ax


def truncate_cmap(cmap, vmin=0.0, vmax=1.0, n=100):
    name = 'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=vmin, b=vmax)
    vals = np.linspace(vmin, vmax, n)
    return colors.LinearSegmentedColormap.from_list(name, vals)


def array_like(a):
    return np.ndim(np.array(a, dtype=object)) > 0


def histogram_bin_edges(X, bins=10, limits=None):
    """Multi-dimensional histogram bin edges."""
    if not array_like(bins):
        bins = X.shape[1] * [bins]
    if not array_like(limits):
        limits = X.shape[1] * [limits]
    return [
        np.histogram_bin_edges(X[:, i], bins[i], limits[i]) 
        for i in range(X.shape[1])
    ]


def plot_image_profiles(
    f,
    x=None,
    y=None,
    ax=None,
    profx=True,
    profy=True,
    kind="step",
    scale=0.12,
    **plot_kws
):
    """Overlay a 1D projection on top of a 2D image.
    Parameters
    ----------
    f : ndarray
        A two-dimensional image.
    x, y : list
        Coordinates of pixel centers.
    ax : matplotlib.pyplt.Axes
        The axis on which to plot.
    profx, profy : bool
        Whether to plot the x/y profile.
    kind : {'step', 'bar', 'line'}
        The type of 1D plot.
    scale : float
        Maximum of the 1D plot relative to the axes limits.
    **plot_kws
        Key word arguments for the 1D plotting function.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if x is None:
        x = np.arange(f.shape[1])
    if y is None:
        y = np.arange(f.shape[0])
    plot_kws.setdefault("lw", 0.75)
    plot_kws.setdefault("color", "white")
    plot_kws.setdefault("drawstyle", "steps-mid")

    def _normalize(profile):
        pmax = np.max(profile)
        if pmax > 0:
            profile = profile / pmax
        return profile

    px, py = [_normalize(np.sum(f, axis=i)) for i in (1, 0)]
    signal_y = y[0] + scale * np.abs(y[-1] - y[0]) * px
    signal_x = x[0] + scale * np.abs(x[-1] - x[0]) * py
    signal_y -= np.min(signal_y) - y[0]
    signal_x -= np.min(signal_x) - x[0]
    for i, (xvals, yvals) in enumerate(zip([x, y], [signal_y, signal_x])):
        if i == 0 and not profx:
            continue
        if i == 1 and not profy:
            continue
        if i:
            xvals, yvals = yvals, xvals
        ax.plot(xvals, yvals, **plot_kws)
    return ax


def plot_image_rms_ellipse(
    f, x=None, y=None, ax=None, level=1.0, center_at_mean=True, **ellipse_kws
):
    """Compute and plot the rms ellipse.
    Parameters
    ----------
    f : ndarray
        A two-dimensional image.
    x, y : list
        Coordinates of pixel centers.
    ax : matplotlib.pyplt.Axes
        The axis on which to plot.
    level : number of list of numbers
        If a number, plot the rms ellipse inflated by the number. If a list
        of numbers, repeat for each number.
    center_at_mean : bool
        Whether to center the ellipse at the image centroid.
    """
    mux = np.average(x, weights=np.sum(f, axis=1))
    muy = np.average(y, weights=np.sum(f, axis=0))
    center = (mux, muy) if center_at_mean else (0.0, 0.0)
    Sigma = psi.cov(f, [x, y])
    return rms_ellipse(Sigma, center, level=level, ax=ax, **ellipse_kws)


def plot_image(
    f,
    x=None,
    y=None,
    ax=None,
    profx=False,
    profy=False,
    prof_kws=None,
    thresh=None,
    thresh_type="abs",
    fill_value=None,
    mask_zero=False,
    floor=None,
    rms_ellipse=False,
    rms_ellipse_kws=None,
    divide_by_max=False,
    return_mesh=False,
    **plot_kws
):
    """Plot a 2D image.
    Parameters
    ----------
    f : ndarray
        A two-dimensional image.
    x, y : list
        Coordinates of pixel centers.
    ax : matplotlib.pyplt.Axes
        The axis on which to plot.
    profx, profy : bool
        Whether to plot the x/y profile.
    prof_kws : dict
        Key words arguments for `image_profiles`.
    thresh : float
        Set elements below this value to zero.
    thresh_type : {'abs', 'frac'}
        If 'frac', `thresh` is a fraction of the maximum element in `f`.
    fill_value : float
        If not None, fills in masked values of `f`.
    mask_zero : bool
        Whether to mask zero values of `f`.
    floor : float
        Add `floor * min(f[f > 0])` to `f`.
    rms_ellipse : bool
        Whether to plot rms ellipse.
    rms_ellipse_kws : dict
        Key word arguments for `image_rms_ellipse`.
    divide_by_max : bool
        Whether to divide the image by its maximum element.
    return_mesh : bool
        Whether to return a mesh from `ax.pcolormesh`.
    **plot_kws
        Key word arguments for `ax.pcolormesh`.
    """
    # plot_kws.setdefault("ec", "None")
    # plot_kws.setdefault("linewidth", 0.0)
    # plot_kws.setdefault("rasterized", True)
    log = "norm" in plot_kws and plot_kws["norm"] == "log"

    f = f.copy()
    if divide_by_max:
        f_max = np.max(f)
        if f_max > 0.0:
            f = f / f_max
    if fill_value is not None:
        f = np.ma.filled(f, fill_value=fill_value)
    if thresh is not None:
        if thresh_type == "frac":
            thresh = thresh * np.max(f)
        f[f < max(1.0e-12, thresh)] = 0
    if mask_zero:
        f = np.ma.masked_less_equal(f, 0)
    if floor is not None:
        _floor = 1.0e-12
        if np.max(f) > 0.0:
            f_min_pos = np.min(f[f > 0])
            floor = floor * f_min_pos
        f = f + floor
    if log:
        if np.any(f == 0):
            f = np.ma.masked_less_equal(f, 0)
        plot_kws['norm'] = colors.LogNorm(vmin=np.min(f), vmax=np.max(f))
    if prof_kws is None:
        prof_kws = dict()
    if x is None:
        x = np.arange(f.shape[0])
    if y is None:
        y = np.arange(f.shape[1])
    if x.ndim == 2:
        x = x.T
    if y.ndim == 2:
        y = y.T
    mesh = ax.pcolormesh(x, y, f.T, **plot_kws)
    if rms_ellipse:
        if rms_ellipse_kws is None:
            rms_ellipse_kws = dict()
        plot_image_rms_ellipse(f, x=x, y=y, ax=ax, **rms_ellipse_kws)
    if profx or profy:
        plot_image_profiles(f, x=x, y=y, ax=ax, profx=profx, profy=profy, **prof_kws)
    if return_mesh:
        return ax, mesh
    else:
        return ax
    
    
def auto_limits(X, sigma=None, pad=0.0, zero_center=False, share=None):
    """Determine axis limits from coordinate array.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinate array for n points in d-dimensional space.
    sigma : float
        If a number is provided, it is used to set the limits relative to
        the standard deviation of the distribution.
    pad : float
        Fractional padding to apply to the limits.
    zero_center : bool
        Whether to center the limits on zero.
    share : tuple[int] or list[tuple[int]]
        Limits are shared betweent the dimensions in each set. For example,
        if `share=(0, 1)`, axis 0 and 1 will share limits. Or if
        `share=[(0, 1), (4, 5)]` axis 0/1 will share limits, and axis 4/5
        will share limits.

    Returns
    -------
    limits : list[tuple]
        The limits [(xmin, xmax), (ymin, ymax), ...].
    """
    if X.ndim == 1:
        X = X[:, None]
    if sigma is None:
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
    else:
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        widths = 2.0 * sigma * stds
        mins = means - 0.5 * widths
        maxs = means + 0.5 * widths
    deltas = 0.5 * np.abs(maxs - mins)
    padding = deltas * pad
    mins = mins - padding
    maxs = maxs + padding
    limits = [(_min, _max) for _min, _max in zip(mins, maxs)]
    if share:
        if np.ndim(share[0]) == 0:
            share = [share]
        for axis in share:
            _min = min([limits[k][0] for k in axis])
            _max = max([limits[k][1] for k in axis])
            for k in axis:
                limits[k] = (_min, _max)
    if zero_center:
        limits = center_limits(limits)
    if len(limits) == 1:
        limits = limits[0]
    return limits


def center_limits(limits):
    """Center limits at zero.

    Example: [(-3, 1), (-4, 5)] --> [(-3, 3), (-5, 5)].

    Parameters
    ----------
    limits : list[tuple]
        A set of limits [(xmin, xmax), (ymin, ymax), ...].

    Returns
    -------
    limits : list[tuple]
        A new set of limits centered at zero [(-x, x), (-y, y), ...].
    """
    mins, maxs = list(zip(*limits))
    maxs = np.max([np.abs(mins), np.abs(maxs)], axis=0)
    return list(zip(-maxs, maxs))


# The following functions are to be called from the simulation. 
# -----------------------------------------------------------------------------
def proj2d(
    X,
    labels=None,
    node=None,
    position=None,    
    axis=(0, 1),
    bins=32,
    limits=None, 
    units=True,
    fig_kws=None,
    text=False,
    colorbar=False,
    colorbar_kws=None,
    **plot_kws
):
    """Plot the 2D projection onto the specified axis."""
    if fig_kws is None:
        fig_kws = dict()
    fig, ax = plt.subplots(**fig_kws)
    if labels is not None:
        ax.set_xlabel(labels[axis[0]])
        ax.set_ylabel(labels[axis[1]])
                
    hist, edges = np.histogramdd(X[:, axis], bins=bins, range=limits)
    centers = [centers_from_edges(e) for e in edges]
    
    ax, mesh = plot_image(hist, x=centers[0], y=centers[1], ax=ax, return_mesh=True, **plot_kws)
    if colorbar:
        if colorbar_kws is None:
            colorbar_kws = dict()
        fig.colorbar(mesh, **colorbar_kws)
    if text:
        if position:
            ax.set_title("s = {:.3f} [m]".format(position))
            
            
def proj2d_three_column(
    X,
    labels=None,
    node=None,
    position=None,    
    axis=[(0, 1), (2, 3), (4, 5)],
    bins='auto',
    limits=None,
    units=False,
    fig_kws=None,
    text=False,
    **plot_kws
):
    """Plot the x-x', y-y', z-dE projections in 1x3 grid.
    
    Parameters
    ----------
    axis : list[tuple]
        The indices to plot in each panel. 
    """
    axis_list = axis
    if limits is None:
        limits = 3 * [limits]
    if fig_kws is None:
        fig_kws = dict()
    fig_kws.setdefault("figsize", (9.0, 2.5))
    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, **fig_kws)
    
    # Set axis labels.
    if labels is not None:
        for ax, axis in zip(axs, axis_list):
            ax.set_xlabel(labels[axis[0]])
            ax.set_ylabel(labels[axis[1]])

    for ax, ind, lims in zip(axs, inds, limits):
        edges = histogram_bin_edges(X[:, ind], bins=bins, limits=lims)
        hist, _ = np.histogramdd(X[:, ind], edges)
        centers = [centers_from_edges(e) for e in edges]  
        ax = plot_image(hist, x=centers[0], y=centers[1], ax=ax, **plot_kws)
        if text:
            # Would like better way to customize this from the PyORBIT script.
            title = ""
            if node is not None:
                title = "{}".format(node)
            if position is not None:
                title = "s={:.2f} ({})".format(position, node)
            ax.set_title(title, fontsize="medium")
            
                
def corner(
    X,
    labels=None,
    node=None,
    position=None,    
    d=4,
    bins=75,
    limits=None,
    fig_kws=None,
    autolim_kws=None,
    diag_kws=None,
    **plot_kws
):
    """Plot all 1D and 2D projections in a corner plot."""
    
    if diag_kws is None:
        diag_kws = dict()
    diag_kws.setdefault("color", "black")
    diag_kws.setdefault("histtype", "step")
    
    if limits is None:
        if autolim_kws is None:
            autolim_kws = dict()
        limits = auto_limits(X, **autolim_kws)
        
    figwidth = 10.0 * (d / 4)
    fig, axs = plt.subplots(ncols=d, nrows=d, figsize=(figwidth, figwidth), sharex=False, sharey=False)
    for i in range(d):
        edges = np.linspace(limits[i][0], limits[i][1], bins + 1)
        axs[i, i].hist(X[:, i], bins=edges, **diag_kws)
    for i in range(d):
        for j in range(i):
            axis = (j, i)
            edges = [np.linspace(limits[k][0], limits[k][1], bins + 1) for k in axis]
            axs[i, j].hist2d(X[:, axis[0]], X[:, axis[1]], bins=edges, **plot_kws)
    
    for ax in axs.ravel():
        for loc in ["top", "right"]:
            ax.spines[loc].set_visible(False)
    
    for i in range(d):
        for j in range(i):
            axs[j, i].axis("off")
    
    for i in range(1, d):
        for j in range(i):
            axs[i, j].set_ylim(limits[i])
    for i in range(d):
        for j in range(i):
            axs[i, j].set_xlim(limits[j])
            
    for i in range(d):
        for j in range(i + 1):
            if (j != 0) or (i == j == 0):
                axs[i, j].set_yticklabels([])
            if i != d - 1:
                axs[i, j].set_xticklabels([])
                
    if labels is not None:
        for i in range(d):
            axs[-1, j].set_xlabel(labels[i])
        for i in range(1, d):
            axs[i, 0].set_ylabel(labels[i])
    return axs
    

class Plotter:
    def __init__(
        self, 
        transform=None, 
        outdir=".", 
        prefix=None, 
        default_save_kws=None, 
        index=0, 
        position=0.0,
        dims=None,
        units=None,
        main_rank=0,
    ):
        self.transform = transform
        self.transforms = []
        self.outdir = outdir
        self.prefix = prefix
        self.functions = []
        self.function_names = []
        self.kws = []
        self.save_kws = []
        self.default_save_kws = default_save_kws
        if self.default_save_kws is None:
            self.default_save_kws  = dict()
        self.index = index
        self.position = position
        self.dims = dims
        self.units = units
        self.labels = None
        if self.dims is not None:
            if self.units is None:
                self.labels = dims
            else:
                self.labels = ["{} [{}]".format(d, u) for d, u in zip(dims, units)]
        self.main_rank = main_rank
        
    def add_function(self, function, transform=None, save_kws=None, name=None, **kws):
        self.functions.append(function)
        self.transforms.append(transform)
        self.kws.append(kws)
        self.save_kws.append(save_kws if save_kws else self.default_save_kws)
        if name is None:
            name = function.__name__
        self.function_names.append(name)
        
    def action(
        self, 
        params_dict=None, 
        index=None,
        node=None,
        position=None,    
        verbose=False
    ):
        _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
        _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
        if _mpi_rank != self.main_rank:
            return
                    
        X = get_bunch_coords(params_dict["bunch"])
        if self.transform is not None:
            X = self.transform(X)
            
        for i, (function, transform) in enumerate(zip(self.functions, self.transforms)):
            if verbose:
                print("Calling {}.".format(self.function_names[i]))
                
            if transform is not None:
                Y = transform(X)
            else:
                Y = X
            
            function(Y, labels=self.labels, node=node, position=position, **self.kws[i])
            
            filename = "{}_{:05.0f}".format(
                self.function_names[i], 
                index if index is not None else self.index
            )
            if node is not None:
                filename = "{}_{}".format(node, filename)
            filename = "fig_{}".format(filename)
            if self.prefix is not None:
                filename = "{}_{}".format(self.prefix, filename)
            filename = os.path.join(self.outdir, filename)
            
            print("Saving figure to file {}".format(filename))
            plt.savefig(filename, **self.save_kws[i])
            plt.close("all")
            
            if position is not None:
                self.position = position
                
        self.index += 1
            

class LinacPlotterNode(BaseLinacNode):
    def __init__(self, name="plotter_node", node_name=None, plotter=None, verbose=True, **kws):
        BaseLinacNode.__init__(self, name)
        self.plotter = plotter
        if self.plotter is None:
            self.plotter = Plotter(**kws)
        self.active = True
        self.node_name = node_name
        self.verbose = verbose
        
    def set_active(self, active):
        self.active = active
        
    def track(self, params_dict):
        if self.active:
            self.plotter.action(params_dict, verbose=self.verbose)
            
    def trackDesign(self, params_dict):
        pass
    
    
class TEAPOTPlotterNode(DriftTEAPOT):
    def __init__(self, name=None, verbose=True, freq=1, plotter=None, **plotter_kws):
        DriftTEAPOT.__init__(self, name)
        self.plotter = plotter
        if self.plotter is None:
            self.plotter = Plotter(**plotter_kws)
        self.name = name
        self.verbose = verbose
        self.freq = freq
        self.turn = 0
        self.active = True
        
    def set_active(self, active):
        self.active = active
        
    def track(self, params_dict):
        if not self.active:
            return
        if self.turn % self.freq == 1:
            self.plotter.action(
                params_dict, 
                index=self.turn,
                node=self.name,
                verbose=self.verbose
            )
        self.turn += 1