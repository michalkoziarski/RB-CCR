import os


if __name__ == '__main__':
    for fold in range(10):
        for classifier_name in ['CART', 'KNN', 'L-SVM', 'R-SVM', 'P-SVM', 'LR', 'NB', 'R-MLP', 'L-MLP']:
            for energy in [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]:
                command = f'sbatch run.sh run_preliminary_energy.py -fold {fold} ' \
                          f'-classifier_name {classifier_name} -energy {energy}'

                os.system(command)
