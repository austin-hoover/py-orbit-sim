"""Test that ORBIT-calculated second-order moments are correct."""
from __future__ import print_function
import os
import sys
from pprint import pprint

import numpy as np

from bunch import Bunch
from bunch import BunchTwissAnalysis


state = np.random.RandomState(15520)
decimals = 3  # for rounding

# Generate covariance matrix.
_cov = np.identity(6)
for (i, j) in [(0, 1), (2, 3), (4, 5), (0, 2)]:
    # Note: x-dE (0-5) or y-dE (2-5) correlations will change the Twiss
    # parameters computed by ORBIT. This is because the ORBIT calculation
    # accounts for dispersion.
    _cov[i, j] = _cov[j, i] = np.random.uniform(-0.8, 0.8)
_cov = 100.0 * _cov

# Generate bunch.
X = np.random.multivariate_normal(np.zeros(6), _cov, size=100000)
bunch = Bunch()
for (x, xp, y, yp, z, dE) in X:
    bunch.addParticle(x, xp, y, yp, z, dE)

# Compute second-order moments
bunch_twiss_analysis = BunchTwissAnalysis()
bunch_twiss_analysis.analyzeBunch(bunch)
order = 2
dispersion_flag = 0
emit_norm_flag = 0
bunch_twiss_analysis.computeBunchMoments(bunch, order, dispersion_flag, emit_norm_flag)
cov = np.zeros((6, 6))
for i in range(6):
    for j in range(6):
        cov[i, j] = bunch_twiss_analysis.getCorrelation(i, j)
np_cov = np.cov(X.T)

# Print comparison
print("Second-order moments")
print("--------------------")
dims = ["x", "x'", "y", "y'", "z", "z'"]
for i in range(6):
    for j in range(6):
        print("<{}{}>:".format(dims[i], dims[j]))
        print("    numpy = {}".format(np_cov[i, j]))
        print("    orbit = {}".format(cov[i, j]))
print()
print("Twiss parameters")
print("----------------")
for i, dim in enumerate(["x", "y", "z"]):
    j = 2 * i
    np_eps = np.sqrt(np_cov[j, j] * np_cov[j + 1, j + 1] - np_cov[j, j + 1] ** 2)
    np_beta = np_cov[j, j] / np_eps
    np_alpha = -np_cov[j, j + 1] / np_eps
    np_gamma = np_cov[j + 1, j + 1] / np_eps
    alpha, beta, gamma, eps = bunch_twiss_analysis.getTwiss(i)
    print("alpha_{}:".format(dim))
    print("    numpy = {}".format(np_alpha))
    print("    orbit = {}".format(alpha))
    print("beta_{}:".format(dim))
    print("    numpy = {}".format(np_beta))
    print("    orbit = {}".format(beta))
    print("gamma_{}:".format(dim))
    print("    numpy = {}".format(np_gamma))
    print("    orbit = {}".format(gamma))
    print("eps_{}:".format(dim))
    print("    numpy = {}".format(np_eps))
    print("    orbit = {}".format(eps))
