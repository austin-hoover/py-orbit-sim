import numpy as np
import pandas as pd


def read_bunch(filename):
    bunch = pd.read_table(filename, comment="%", sep="\s+", index_col=None)    
    columns = ["x", "xp", "y", "yp", "z", "dE"]
    if bunch.shape[1] > 6:
        if bunch.shape[1] == 7 or bunch.shape == 13:
            columns += ["index"]
        if bunch.shape[1] > 7:
            columns += ["Jx", "phasex", "nux", "nuy", "Jy", "phasey"]
    bunch.columns = columns
    columns = [c for c in columns if c not in ["phasex", "Jx", "phasey", "Jy"]]
    bunch = bunch.loc[:, columns]
    bunch.iloc[:, :4] *= 1000.0  # [m, rad] -> [mm, mrad]
    bunch.iloc[:, 5] *= 1000.0  # [GeV] -> [MeV]
    return bunch