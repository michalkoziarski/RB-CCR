import os


if __name__ == '__main__':
    for fold in range(10):
        for classifier_name in ['cart', 'knn', 'svm', 'rsvm', 'psvm', 'lr', 'nb', 'mlp', 'lmlp']:
            for resampler_name in ['rb-ccr-minority', 'rb-ccr-majority', 'rb-ccr-relative']:
                command = f'sbatch run.sh run_preliminary_scoring.py -fold {fold} ' \
                          f'-classifier_name {classifier_name} -resampler_name {resampler_name}'

                os.system(command)
