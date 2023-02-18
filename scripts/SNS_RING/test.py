from __future__ import print_function

from bunch import Bunch
from orbit.utils.consts import mass_proton

from SNS_RING import SNS_RING


ring = SNS_RING(
    nominal_n_inj_turns=1000,
    nominal_intensity=1.5e14,
    nominal_bunch_length_frac=(50.0 / 64.0),
)
ring.readMADX(
    "/home/46h/repo/accelerator-models/SNS/RING/SNS_RING_nux6.18_nuy6.18.lat", 
    "rnginj",
)
ring.initialize()
ring.init_bunch(mass=mass_proton, kin_energy=1.0)

# Injection kickers
# [...]

# ring.add_inj_node()
ring.add_foil_node()

ring.add_inj_chicane_aperture_displacement_nodes()

## Apertures need to be added as child nodes, but Jeff's benchmarks script
## selects the parent nodes by index; does not work with standard MADX
## output lattice.
# ring.add_aperture_nodes()

ring.add_collimator_nodes()

ring.add_rf_harmonic_nodes(
    RF1=dict(phase=0.0, hnum=1.0, voltage=+2.0e-6),
    RF2=dict(phase=0.0, hnum=2.0, voltage=-4.0e-6)
)

ring.add_longitudinal_impedance_node(
    n_macros_min=1000,
    n_bins=128,
    position=124.0,
    ZL_Ekicker=None,  # read from file
    ZL_RF=None,  # read from file
)

ring.add_transverse_impedance_node(
    n_macros_min=1000,
    n_bins=64,
    use_x=0,
    use_y=1,
    position=124.0,
    alpha_x=0.0,
    alpha_y=-0.004,
    beta_x=10.191,
    beta_y=10.447,
    q_x=6.21991,
    q_y=6.20936,
)

ring.add_longitudinal_space_charge_node(
    b_a=(10.0 / 3.0),
    n_macros_min=1000,
    use=1,
    n_bins=64,
    position=124.0,
)

ring.add_transverse_space_charge_nodes(
    n_macros_min=1000,
    size_x=128,
    size_y=128,
    size_z=64,
    path_length_min=1.0e-8,
    n_boundary_points=128,
    n_free_space_modes=32,
    r_boundary=0.220,
    kind="slicebyslice",
)

ring.add_tune_diagnostics_node(
    position=51.1921,
    beta_x=9.19025, 
    alpha_x=-1.78574, 
    eta_x=-0.000143012, 
    etap_x=-2.26233e-05, 
    beta_y=8.66549, 
    alpha_y=0.538244,
)

ring.add_moments_diagnostics_node(order=4, position=0.0)