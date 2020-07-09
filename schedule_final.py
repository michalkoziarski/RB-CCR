import os


if __name__ == '__main__':
    for fold in range(10):
        for classifier_name in ['cart', 'knn', 'svm', 'rsvm', 'psvm', 'lr', 'nb', 'mlp', 'lmlp']:
            command = f'sbatch run.sh run_final.py -fold {fold} -classifier_name {classifier_name}'

            os.system(command)
