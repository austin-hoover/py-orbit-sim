"""Within-simulation plotting routines.

(Uses python2-compatible matplotlib.)
"""
import os
import sys

import matplotlib
matplotlib.use('agg')
from matplotlib import colors
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np


DIMS = ["x", "xp", "y", "yp", "z", "dE"]
UNITS = ["m", "rad", "m", "rad", "m", "GeV"]


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


def plot1d(x, y, ax=None, flipxy=False, kind="step", **kws):
    """Convenience function for one-dimensional line/step/bar plots."""
    funcs = {
        "line": ax.plot,
        "bar": ax.bar,
        "step": ax.plot,
    }
    if kind == "step":
        kws.setdefault("drawstyle", "steps-mid")
    if flipxy:
        x, y = y, x
        funcs["bar"] = ax.barh
    return funcs[kind](x, y, **kws)


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


def histogram_bin_edges(X, bins=10, limits=None):
    n = X.shape[1]
    if type(bins) not in [list, tuple]:
        bins = n * [bins]
    if type(limits) not in [list, tuple]:
        limits = n * [limits] 
    return [np.histogram_bin_edges(X[:, i], bins[i], limits[i]) for i in range(n)]


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
        ax = plot1d(xvals, yvals, ax=ax, flipxy=i, kind=kind, **plot_kws)
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
    **plot_kws,
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
    plot_kws.setdefault("ec", "None")
    plot_kws.setdefault("linewidth", 0.0)
    plot_kws.setdefault("rasterized", True)
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
        plot_image_rms_ellipsef, x=x, y=y, ax=ax, **rms_ellipse_kws)
    if profx or profy:
        plot_image_profiles(f, x=x, y=y, ax=ax, profx=profx, profy=profy, **prof_kws)
    if return_mesh:
        return ax, mesh
    else:
        return ax


# The following functions are to be called within the simulation. 
# -----------------------------------------------------------------------------
def proj2d(
    X,
    info=None, 
    axis=(0, 1),
    bins='auto', 
    limits=None, 
    units=True,
    fig_kws=None,
    text=None,
    **plot_kws
):
    """Plot the 2D projection onto the specified axis."""
    if fig_kws is None:
        fig_kws = dict()
    fig, ax = plt.subplots(**fig_kws)
    labels = True
    if labels:
        if units:
            ax.set_xlabel("{} [{}]".format(DIMS[axis[0]], UNITS[axis[0]]))
            ax.set_ylabel("{} [{}]".format(DIMS[axis[1]], UNITS[axis[1]]))
        else:
            ax.set_xlabel("{}".format(DIMS[axis[0]]))
            ax.set_ylabel("{}".format(DIMS[axis[1]]))
    
    edges = histogram_bin_edges(X[:, axis], bins=bins, limits=limits)
    hist, _ = np.histogramdd(X[:, axis], edges)
    centers = [centers_from_edges(e) for e in edges]
    ax = plot_image(hist, x=centers[0], y=centers[1], ax=ax, **plot_kws)
    
    if text is not None:
        if 's' in info:
            ax.set_title('s = {:.3f} [m]'.format(info['s']))
            
            
def proj2d_three_column(
    X,
    info=None,
    axis=[(0, 1), (2, 3), (4, 5)],
    bins='auto',
    limits=None,
    units=False,
    fig_kws=None,
    text=None,
    **plot_kws
):
    """Plot the x-x', y-y', z-dE projections in 1x3 grid.
    
    Parameters
    ----------
    axis : list[tuple]
        The indices to plot in each panel. 
    """
    inds = axis
    if limits is None:
        limits = 3 * [limits]
    if fig_kws is None:
        fig_kws = dict()
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, **fig_kws)
    labels = True
    if labels:
        for ax, (i, j) in zip(axes, inds):
            if units:
                ax.set_xlabel("{} [{}]".format(DIMS[i], UNITS[i]))
                ax.set_ylabel("{} [{}]".format(DIMS[j], UNITS[j]))
            else:
                ax.set_xlabel("{}".format(DIMS[i]))
                ax.set_ylabel("{}".format(DIMS[j]))
        
    for ax, ind, lims in zip(axes, inds, limits):
        edges = histogram_bin_edges(X[:, ind], bins=bins, limits=lims)
        hist, _ = np.histogramdd(X[:, ind], edges)
        centers = [centers_from_edges(e) for e in edges]  
        ax = plot_image(hist, x=centers[0], y=centers[1], ax=ax, **plot_kws)
        if text is not None:
            if 's' in info:
                ax.set_title('s = {:.3f} [m]'.format(info['s']))

                
def corner():
    raise NotImplementedError
        

class Plotter:
    def __init__(self, transform=None, path='.', default_save_kws=None):
        self.transform = transform
        self.path = path
        self.funcs = []
        self.kws = []
        self.save_kws = []
        self.default_save_kws = default_save_kws
        if self.default_save_kws is None:
            self.default_save_kws  = dict()
        
    def add_func(self, func, save_kws=None, name=None, **kws):
        self.funcs.append(func)
        self.kws.append(kws)
        self.names.append(name if name else 'plot_{}'.format(len(self.funcs)))
        self.save_kws.append(save_kws if save_kws else self.default_save_kws)
        
    def plot(self, X, info=None, verbose=False):
        if self.transform is not None:
            X = self.transform(X)
        for i in range(len(self.funcs)):
            if verbose:
                print("Calling '{}' ({}).".format(self.names[i], self.func.__name__))
            self.funcs[i](X, info=info, **self.kws[i])
            filename = self.names[i]
            if 'step' in info:
                filename = '{}_{:03}'.format(filename, info['step'])
            if 'node' in info:
                filename = '{}_{}'.format(filename, info['node'])
            filename = os.path.join(self.path, filename)
            plt.savefig(os.path.join(self.path, filename, **self.save_kws[i])
            plt.close()