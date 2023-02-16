import numpy as np
import pandas as pd


def read_bunch(filename, dataframe=False):
    """Read bunch text file.
    
    TODO
    ----
    * Option to read additional bunch attributes/columns (tune, etc.)
    """
    dims = ["x", "xp", "y", "yp", "z", "dE"]
    df = pd.read_table(filename, sep=" ", skiprows=14, usecols=range(len(dims)), names=dims)
    if dataframe:
        return df
    return df.values