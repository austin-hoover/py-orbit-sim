# PyORBIT scripts

This repository contains PyORBIT scripts for accelerator physics studies at the SNS.

Run scripts from the root directory. Example for two processors: 

```shell
cd /py-orbit
source setupEnvironment.sh
cd ../py-orbit-sim
mpirun -np 2 ${ORBIT_ROOT}/bin/pyORBIT scripts/mpi/test_argparse.py --d="hi"
```

Data is written to `/scripts/data/`.