from __future__ import print_function
import os

import numpy as np


folder = "/home/46h/sim_data/SNS_BTF/sim_RFQ_HZ04_reverse/2023-03-09/"
filename = os.path.join(
    folder,
    "230309125322-sim_RFQ_HZ04_reverse_bunch_RFQ.dat",
)
print("Loading file {}".format(filename))
X = np.loadtxt(filename, comments="%")
print("Decorrelating")
for i in range(0, X.shape[1], 2):
    idx = np.random.permutation(np.arange(X.shape[0]))
    X[:, i : i + 2] = X[idx, i : i + 2]
filename = os.path.join(
    folder,
    "230309125322-sim_RFQ_HZ04_reverse_bunch_RFQ_decorrelated.dat",
)
print("Saving decorrelated bunch to file {}".format(filename))
np.savetxt(os.path.join(folder, filename), X)