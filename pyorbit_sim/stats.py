import numpy as np


def twiss_2x2(Sigma):
    """RMS Twiss parameters from 2 x 2 covariance matrix.

    Parameters
    ----------
    Sigma : ndaray, shape (2, 2)
        The covariance matrix for position u and momentum u' [[<uu>, <uu'>], [<uu'>, <u'u'>]].

    Returns
    -------
    alpha : float
        The alpha parameter (-<uu'> / sqrt(<uu><u'u'> - <uu'>^2)).
    beta : float
        The beta parameter (<uu> / sqrt(<uu><u'u'> - <uu'>^2)).
    """
    eps = emittance_2x2(Sigma)
    beta = Sigma[0, 0] / eps
    alpha = -Sigma[0, 1] / eps
    return alpha, beta


def emittance_2x2(Sigma):
    """RMS emittance from u-u' covariance matrix.

    Parameters
    ----------
    Sigma : ndaray, shape (2, 2)
        The covariance matrix for position u and momentum u' [[<uu>, <uu'>], [<uu'>, <u'u'>]].

    Returns
    -------
    float
        The RMS emittance (sqrt(<uu><u'u'> - <uu'>^2)).
    """
    return np.sqrt(np.linalg.det(Sigma))


def apparent_emittances(Sigma):
    """Compute rms apparent emittances from 2n x 2n covariance matrix.

    Parameters
    ----------
    Sigma : ndarray, shape (2n, 2n)
        A covariance matrix. (Dimensions ordered {x, x', y, y', ...}.)

    Returns
    -------
    eps_x, eps_y, eps_z, ... : float
        The emittance in each phase-plane (eps_x, eps_y, eps_z, ...)
    """
    emittances = []
    for i in range(0, Sigma.shape[0], 2):
        emittances.append(emittance_2x2(Sigma[i : i + 2, i : i + 2]))
    if len(emittances) == 1:
        emittances = emittances[0]
    return emittances


def intrinsic_emittances(Sigma):
    """Compute rms intrinsic emittances from 4 x 4 covariance matrix.

    Parameters
    ----------
    Sigma : ndarray, shape (4, 4)
        A covariance matrix. (Dimensions ordered {x, x', y, y', ...}.)

    Returns
    -------
    eps_1, eps_2 : float
        The intrinsic emittances (eps_4D = sqrt(det(Sigma)) = eps_1 * eps_2).
    """
    U = np.array([
        [+0.0, +1.0, +0.0, +0.0], 
        [-1.0, +0.0, +0.0, +0.0], 
        [+0.0, +0.0, +0.0, +1.0], 
        [+0.0, +0.0, -1.0, +0.0],
    ])
    tr_SU2 = np.trace(np.linalg.matrix_power(np.matmul(Sigma, U), 2))
    det_S = np.linalg.det(Sigma)
    eps_1 = 0.5 * np.sqrt(-tr_SU2 + np.sqrt(tr_SU2**2 - 16.0 * det_S))
    eps_2 = 0.5 * np.sqrt(-tr_SU2 - np.sqrt(tr_SU2**2 - 16.0 * det_S))
    return eps_1, eps_2


def twiss(Sigma):
    """Compute rms Twiss parameters from 2n x 2n covariance matrix.

    Parameters
    ----------
    Sigma : ndarray, shape (2n, 2n)
        A covariance matrix. (Dimensions ordered {x, x', y, y', ...}.)

    Returns
    -------
    alpha_x, beta_x, alpha_y, beta_y, alpha_z, beta_z, ... : float
        The Twiss parameters in each plane.
    """
    n = Sigma.shape[0] // 2
    params = []
    for i in range(n):
        j = i * 2
        params.extend(twiss_2x2(Sigma[j : j + 2, j : j + 2]))
    return params