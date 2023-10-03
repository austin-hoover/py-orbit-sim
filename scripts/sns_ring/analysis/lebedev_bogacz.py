import numpy as np


def unit_symplectic_matrix(d=2):
    U = np.zeros((d, d))
    for i in range(0, d, 2):
        U[i : i + 2, i : i + 2] = [[0.0, 1.0], [-1.0, 0.0]]
    return U


def normalize_eigenvectors(eigenvectors):
    d = eigenvectors.shape[0]
    U = unit_symplectic_matrix(d)
    for i in range(0, d, 2):
        v = eigenvectors[:, i]
        val = np.linalg.multi_dot([np.conj(v), U, v]).imag
        if val > 0.0:
            (eigenvectors[:, i], eigenvectors[:, i + 1]) = (eigenvectors[:, i + 1], eigenvectors[:, i])
        eigenvectors[:, i : i + 2] *= np.sqrt(2.0 / np.abs(val))
    return eigenvectors


def normalization_matrix_from_eigenvectors(eigenvectors):
    V = np.zeros(eigenvectors.shape)
    for i in range(0, V.shape[1], 2):
        V[:, i] = eigenvectors[:, i].real
        V[:, i + 1] = (1.0j * eigenvectors[:, i]).real
    return V


def get_bunch_normalization_matrix(X):
    d = 4
    Sigma = np.cov(X[:, :d].T)
    U = unit_symplectic_matrix(d)
    SU = np.matmul(Sigma, U)
    eigenvalues, eigenvectors = np.linalg.eig(SU)
    eigenvectors = normalize_eigenvectors(eigenvectors)
    W = normalization_matrix_from_eigenvectors(eigenvectors)
    return W