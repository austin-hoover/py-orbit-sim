from __future__ import print_function
import os

import numpy as np
import pandas as pd


def read_orbit_bunch_file(filename, dims=None, dframe=False):
    names = ["x", "xp", "y", "yp", "z", "dE"]
    cols = list(range(6))
    if dims is not None:
        cols = [d if type(d) is int else names.index(d) for d in dims]
    names = [names[c] for c in cols]
    df = pd.read_table(filename, sep=" ", skiprows=14, usecols=cols, names=names)
    # Convert to mm, mrad, keV
    for col in ["x", "xp", "y", "yp", "z"]:
        if col in df.columns:
            df[col] *= 1e3
    if "dE" in df.columns:
        df["dE"] *= 1e6
    if dframe:
        return df
    return df.values


# The following three functions allow saving/loading of ragged arrays. An
# example use case is to save the turn-by-turn coordinates of a bunch that
# has a different number of particles on each turn.
# Source: https://tonysyu.github.io/ragged-arrays.html#.YKVwQy9h3OR
def stack_ragged(array_list, axis=0):
    """Stacks list of arrays along first axis.

    Example: (25, 4) + (75, 4) -> (100, 4). It also returns the indices at
    which to split the stacked array to regain the original list of arrays.
    """
    lengths = [np.shape(array)[axis] for array in array_list]
    idx = np.cumsum(lengths[:-1])
    stacked = np.concatenate(array_list, axis=axis)
    return stacked, idx


def save_stacked_array(filename, array_list, axis=0):
    """Save list of ragged arrays as single stacked array. The index from
    `stack_ragged` is also saved."""
    stacked, idx = stack_ragged(array_list, axis=axis)
    np.savez(filename, stacked_array=stacked, stacked_index=idx)


def load_stacked_arrays(filename, axis=0):
    """ "Load stacked ragged array from .npz file as list of arrays."""
    npz_file = np.load(filename)
    idx = npz_file["stacked_index"]
    stacked = npz_file["stacked_array"]
    return np.split(stacked, idx, axis=axis)
