from bunch import bunch
from orbit.bunch_generators import GaussDist3D
from orbit.bunch_generators import KVDist3D
from orbit.bunch_generators import TwissContainer
from orbit.bunch_generators import WaterBagDist3D
import orbit_mpi

import pyorbit_sim


def get_design_bunch(
    dist=WaterBagDist3D,
    n_parts=10000, 
    mass=0.939294,  # [GeV / c^2]
    kin_energy=0.0025,  # [GeV]
    alpha_x=-1.9620,
    alpha_y=1.7681,
    alpha_z=-0.0196,
    beta_x=0.1831,
    beta_y=0.1620,
    beta_z=0.5844,
    eps_x=0.21e-6,
    eps_y=0.21e-6,
    eps_z=0.24153e-6,
    bunch=None,
    verbose=True,
):    
    """Return bunch with design Twiss parameters. 
    
    Mass and kinetic energy are not changed.
    """
    _mpi_comm = orbit_mpi.mpi_comm.MPI_COMM_WORLD
    _mpi_rank = orbit_mpi.MPI_Comm_rank(_mpi_comm)
        
    gamma = (mass + kin_energy) / mass
    beta = math.sqrt(gamma * gamma - 1.0) / gamma
    eps_x = eps_x / (beta * gamma)  # [m * rad]
    eps_y = eps_y / (beta * gamma)  # [m * rad]
    eps_z = eps_z / (beta * gamma**3)  # [m * rad]
    eps_z = eps_z * gamma**3 * beta**2 * mass  # [m * GeV]
    beta_z = beta_z / (gamma**3 * beta**2 * mass)    
    if _mpi_rank == 0:
        print("Generating bunch from design Twiss parameters and {} generator.".format(dist))    
    bunch = pyorbit_sim.bunch_utils.generate(
        dist=dist(
            twissX=TwissContainer(alpha_x, beta_x, eps_x),
            twissY=TwissContainer(alpha_y, beta_y, eps_y),
            twissZ=TwissContainer(alpha_z, beta_z, eps_z),
        ),
        bunch=bunch,
        n_parts=n_parts,
        verbose=verbose,
    )    
    bunch.mass(mass)
    bunch.getSyncParticle().kinEnergy(kin_energy)
    return bunch