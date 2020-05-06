import datasets
import os


if __name__ == '__main__':
    for dataset_name in datasets.names('final'):
        for fold in range(10):
            for classifier_name in ['cart', 'knn', 'svm', 'lr', 'nb', 'mlp']:
                for energy in [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]:
                    for gamma in [0.5, 1.0, 2.5, 5.0, 10.0]:
                        command = f'sbatch run.sh run_prelim_param_parallel.py -classifier_name {classifier_name} ' \
                                  f'-dataset_name {dataset_name} -fold {fold} -energy {energy} -gamma {gamma}'

                        os.system(command)
