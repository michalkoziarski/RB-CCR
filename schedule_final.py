import os


if __name__ == '__main__':
    for fold in range(10):
        for classifier_name in ['CART', 'KNN', 'L-SVM', 'R-SVM', 'P-SVM', 'LR', 'NB', 'R-MLP', 'L-MLP']:
            command = f'sbatch run.sh run_final.py -fold {fold} -classifier_name {classifier_name}'

            os.system(command)
