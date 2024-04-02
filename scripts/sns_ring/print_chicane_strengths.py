from orbit.teapot import teapot

lattice = teapot.TEAPOT_Lattice()
lattice.readMAD("./data_input/sns_ring_nux6.24_nuy6.18_mad.lattice", "RINGINJ")
lattice.initialize()


chicane_node_names = [
    "DH_A10",
    "DH_A11A",
    "DH_A11B",
    "DH_A12",
    "DH_A13",
]

for name in chicane_node_names:
    node = lattice.getNodeForName(name)
    print(node.getName())




