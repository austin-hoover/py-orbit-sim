# PyORBIT scripts

This repository contains PyORBIT scripts for accelerator physics studies at the SNS.

Run scripts from the root directory with mpirun. Example for two processors: 

```shell
cd /py-orbit
source setupEnvironment.sh
cd ../py-orbit-sim
mpirun -np 2 ${ORBIT_ROOT}/bin/pyORBIT scripts/mpi/test_argparse.py --d="hi"
```

2024-06-05: moving to PyORBIT3: https://github.com/austin-hoover/pyorbit-sim
