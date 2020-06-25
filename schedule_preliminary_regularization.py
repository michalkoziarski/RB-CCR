import os


if __name__ == '__main__':
    classifier_names = []

    for c in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        classifier_names.append(f'svm[{c}]')

    for a in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]:
        classifier_names.append(f'mlp[{a}]')

    for fold in range(10):
        for classifier_name in classifier_names:
            for regions in ['L', 'E', 'H', 'LEH']:
                command = f'sbatch run.sh run_preliminary_regularization.py ' \
                          f'-classifier_name {classifier_name} ' \
                          f'-fold {fold} -regions {regions}'

                os.system(command)
