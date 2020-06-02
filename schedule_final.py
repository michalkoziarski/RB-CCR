import datasets
import os


if __name__ == '__main__':
    for minority_training_size in [-1, 5, 10, 15, 20, 30]:
        for dataset_name in datasets.names():
            for fold in range(10):
                for classifier_name in ['cart', 'knn', 'svm', 'lr', 'nb', 'mlp']:
                        command = f'sbatch run.sh run_final.py -minority_training_size {minority_training_size} ' \
                                  f'-dataset_name {dataset_name} -fold {fold} -classifier_name {classifier_name}'

                        os.system(command)
