import os


if __name__ == '__main__':
    for fold in range(10):
        for classifier_name in ['cart', 'knn', 'svm', 'lr', 'nb', 'mlp']:
            for regions in ['L', 'E', 'H', 'LE', 'LH', 'EH', 'LEH']:
                for region_type in ['safe', 'borderline', 'rare', 'outlier']:
                    command = f'sbatch run.sh run_preliminary_regions_type.py -classifier_name {classifier_name} ' \
                              f'-fold {fold} -regions {regions} -type {region_type}'

                    print(command)#os.system(command)
