import numpy as np


def norm_xpx_ypy_zpz(X, scale_emittance=False):
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