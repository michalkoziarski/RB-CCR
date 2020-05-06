import datasets
import os


if __name__ == '__main__':
    for dataset_name in datasets.names('final'):
        for fold in range(10):
            for classifier_name in ['cart', 'knn', 'svm', 'lr', 'nb', 'mlp']:
                for regions in ['L', 'E', 'H', 'LE', 'LH', 'EH', 'LEH']:
                    command = f'sbatch run.sh run_prelim_reg_parallel.py -classifier_name {classifier_name} ' \
                              f'-dataset_name {dataset_name} -fold {fold} -regions {regions}'

                    os.system(command)
