import numpy as np

from bunch import Bunch
from orbit.bunch_generators import GaussDist2D
from orbit.bunch_generators import KVDist2D
from orbit.bunch_generators import WaterBagDist2D

import pyorbit_sim



class BunchGenerator:
    def __init__(self, kind="waterbag"):
        assert kind in KINDS        
        dists = {
            "danilov22": DanilovDist22,
            "gaussian": GaussDist2D,
            "kv": KVDist2D, 
            "waterbag": WaterBagDist2D, 
        }
        self.dist = dists[kind]
                
    def sample(self, n):
        return pyorbit_sim.bunch_utils.generate(dist=self.dist, n=n, bunch=None, verbose=True)
        
    