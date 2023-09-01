# PyORBIT scripts

This repository contains PyORBIT scripts for accelerator physics studies at the SNS.

Run scripts from the root directory with mpirun. Example for two processors:``./START.sh scripts/mpi/test_argparse.py 2 --arg="hi"`` or ``mpirun -np 2 ${ORBIT_ROOT}/bin/pyORBIT scripts/mpi/test_argparse.py --arg="hi"``.


Note that the `START.sh` script is different than in the PyORBIT examples repo to allow passing command line arguments. 