import os


if __name__ == '__main__':
    for fold in range(10):
        for classifier_name in ['svm', 'lr', 'mlp']:
            for regions in ['L', 'E', 'H', 'LE', 'LH', 'EH', 'LEH']:
                command = f'sbatch run.sh run_preliminary_regions_proba.py -classifier_name {classifier_name} ' \
                          f'-fold {fold} -regions {regions}'

                os.system(command)
