import os


if __name__ == '__main__':
    for fold in range(10):
        for degree in range(1, 10):
            classifier_name = f'svm({degree})'

            for regions in ['L', 'E', 'H', 'LEH']:
                command = f'sbatch run.sh run_preliminary_regions_svm_degree.py ' \
                          f'-classifier_name {classifier_name} ' \
                          f'-fold {fold} -regions {regions}'

                os.system(command)
