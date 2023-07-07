import numpy as np


def get_radii(X):
    return np.linalg.norm(X, axis=1)


def get_ellipsoid_radii(X):
    Sigma_inv = np.linalg.inv(np.cov(X.T))
    func = lambda point: np.sqrt(np.linalg.multi_dot([point.T, Sigma_inv, point]))
    return transform(X, func)


def get_enclosing_sphere_radius(X, axis=None, fraction=1.0):
    """Scales sphere until it contains some fraction of points.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    axis : tuple
        The distribution is projected onto this axis before proceeding. The
        ellipsoid is defined in this subspace.
    fraction : float
        Fraction of points in sphere.

    Returns
    -------
    radius : float
        The sphere radius.
    """
    radii = np.sort(get_radii(X[:, axis]))
    index = int(np.round(X.shape[0] * fraction)) - 1
    return radii[index]


def get_enclosing_ellipsoid_radius(X, axis=None, fraction=1.0):
    """Scale the rms ellipsoid until it contains some fraction of points.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    axis : tuple
        The distribution is projected onto this axis before proceeding. The
        ellipsoid is defined in this subspace.
    fraction : float
        Fraction of points enclosed.

    Returns
    -------
    float
        The ellipsoid "radius" (x^T Sigma^-1 x) relative to the rms ellipsoid.
    """
    radii = np.sort(get_ellipsoid_radii(X[:, axis]))
    index = int(np.round(X.shape[0] * fraction)) - 1
    return radii[index]


def transform(X, func=None, **kws):
    """Apply a nonlinear transformation.

    This function just calls `np.apply_along_axis`.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    function : callable
        Function applied to each point in X. Call signature is
        `function(point, **kws)` where `point` is an n-dimensional
        point given by one row of `X`.
    **kws
        Key word arguments for

    Returns
    -------
    ndarray, shape (n, d)
        The transformed distribution.
    """
    return np.apply_along_axis(lambda point: func(point, **kws), 1, X)


def transform_linear(X, M):
    """Apply a linear transformation.

    This function just calls `np.apply_along_axis`.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    M : ndarray, shape (d, d)
        A linear transfer matrix.

    Returns
    -------
    ndarray, shape (n, d)
        The transformed distribution.
    """
    func = lambda point: np.matmul(M, point)
    return transform(X, lambda point: np.matmul(M, point))


def norm_xxp_yyp_zzp(X, scale_emittance=False):
    Sigma = np.cov(X.T)
    Xn = np.zeros(X.shape)
    for i in range(0, X.shape[1], 2):
        _Sigma = Sigma[i : i + 2, i : i + 2]
        eps = np.sqrt(np.linalg.det(_Sigma))
        alpha = -_Sigma[0, 1] / eps
        beta = _Sigma[0, 0] / eps
        Xn[:, i] = X[:, i] / np.sqrt(beta)
        Xn[:, i + 1] = (np.sqrt(beta) * X[:, i + 1]) + (alpha * X[:, i] / np.sqrt(beta))
        if scale_emittance:
            Xn[:, i : i + 2] = Xn[:, i : i + 2] / np.sqrt(eps)
    return Xn


def slice_planar(X, axis=None, center=None, width=None, limits=None):
    """Return points within a planar slice.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    axis : tuple
        Slice axes. For example, (0, 1) will slice along the first and
        second axes of the array.
    center : ndarray, shape (n,)
        The center of the box.
    width : ndarray, shape (d,)
        The width of the box along each axis.
    limits : ndarray, shape (d, 2)
        The (min, max) along each axis. Overrides `center` and `edges` if provided.

    Returns
    -------
    ndarray, shape (?, n)
        The points within the box.
    """
    n, d = X.shape
    if not array_like(axis):
        axis = (axis,)
    if limits is None:
        if not array_like(center):
            center = np.full(d, center)
        if not array_like(width):
            width = np.full(d, width)
        center = np.array(center)
        width = np.array(width)
        limits = list(zip(center - 0.5 * width, center + 0.5 * width))  
    limits = np.array(limits)
    if limits.ndim == 0:
        limits = limits[None, :]
    conditions = []
    for j, (umin, umax) in zip(axis, limits):
        conditions.append(X[:, j] > umin)
        conditions.append(X[:, j] < umax)
    idx = np.logical_and.reduce(conditions)
    return X[idx, :]


def slice_sphere(X, axis=None, rmin=0.0, rmax=None):
    """Return points within a spherical shell slice.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    axis : tuple
        The subspace in which to define the sphere.
    rmin, rmax : float
        Inner/outer radius of spherical shell.

    Returns
    -------
    ndarray, shape (?, d)
        The points within the sphere.
    """
    if rmax is None:
        rmax = np.inf
    radii = get_radii(X[:, axis])
    idx = np.logical_and(radii > rmin, radii < rmax)
    return X[idx, :]


def slice_ellipsoid(X, axis=None, rmin=0.0, rmax=None):
    """Return points within an ellipsoidal shell slice.

    The ellipsoid is defined by the covariance matrix of the
    distribution.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    axis : tuple
        The subspace in which to define the ellipsoid.
    rmin, rmax : list[float]
        Min/max "radius" (x^T Sigma^-1 x). relative to covariance matrix.

    Returns
    -------
    ndarray, shape (?, d)
        Points within the shell.
    """
    if rmax is None:
        rmax = np.inf
    radii = get_ellipsoid_radii(X[:, axis])
    idx = np.logical_and(rmin < radii, radii < rmax)
    return X[idx, :]


def slice_contour(X, axis=None, lmin=0.0, lmax=1.0, interp=True, **hist_kws):
    """Return points within a contour shell slice.

    The slice is defined by the density contours in the subspace defined by
    `axis`.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    axis : tuple
        The subspace in which to define the density contours.
    lmin, lmax : list[float]
        If `f` is the density in the subspace defined by `axis`, then we select
        points where lmin <= f / max(f) <= lmax.
    interp : bool
        If True, compute the histogram, then interpolate and evaluate the
        resulting function at each point in `X`. Otherwise we keep track
        of the indices in which each point lands when it is binned,
        and accept the point if it's bin has a value within fmin and fmax.
        The latter is a lot slower.

    Returns
    -------
    ndarray, shape (?, d)
        Points within the shell.
    """
    _X = X[:, axis]
    hist, edges = histogram(_X, **hist_kws)
    hist = hist / np.max(hist)
    centers = [0.5 * (e[:-1] + e[1:]) for e in edges]
    if interp:
        fint = scipy.interpolate.RegularGridInterpolator(
            centers,
            hist,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        values = fint(_X)
        idx = np.logical_and(lmin <= values, values <= lmax)
    else:
        valid_indices = np.vstack(
            np.where(np.logical_and(lmin <= hist, hist <= lmax))
        ).T
        indices = np.vstack(
            [np.digitize(_X[:, k], edges[k]) for k in range(_X.shape[1])]
        ).T
        idx = []
        for i in range(len(indices)):
            if indices[i].tolist() in valid_indices.tolist():
                idx.append(i)
    return X[idx, :]