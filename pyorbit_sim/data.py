import numpy as np
import pandas as pd


def read_orbit_bunch_file(filename, dframe=False):
    dims = ["x", "xp", "y", "yp", "z", "dE"]
    df = pd.read_table(filename, sep=" ", skiprows=14, usecols=range(len(dims)), names=dims)
    if dframe:
        return df
    return df.values