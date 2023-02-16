from __future__ import print_function

import numpy as np

from bunch import Bunch
from orbit.injection.distributions import JohoTransversePaint


bunch = Bunch()
bunch.mass(0.938)
sync_part = bunch.getSyncParticle()
sync_part.kinEnergy(1.0)


dist = JohoTransversePaint(sync_part=sync_part, exp=3)
coords = dist.getCoordinates()

print(coords)