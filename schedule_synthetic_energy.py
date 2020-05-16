import numpy as np
import os


if __name__ == '__main__':
    for fold in range(10):
        for imbalance_ratio in [2.0, 5.0, 10.0]:
            for energy in np.arange(2.0, 52.0, 2.0):
                command = f'sbatch run.sh run_synthetic_energy.py ' \
                          f'-fold {fold} -energy {energy} ' \
                          f'-imbalance_ratio {imbalance_ratio}'

                os.system(command)
