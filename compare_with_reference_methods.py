import numpy as np
import pandas as pd

from analyse_regions import test_friedman_shaffer
from collections import OrderedDict
from merge import RESULTS_PATH


RESAMPLERS = ['None', 'SMOTE', 'Bord', 'NCL', 'SMOTE+TL', 'SMOTE+EN', 'RB-CCR-CV']
CLASSIFIERS = ['CART', 'KNN', 'L-SVM', 'R-SVM', 'P-SVM', 'LR', 'NB', 'R-MLP', 'L-MLP']
METRICS = ['Precision', 'Recall', 'Specificity', 'AUC', 'F-measure', 'G-mean']
P_VALUE = 0.10


def load_final_dict(classifier, metric):
    df = pd.read_csv(RESULTS_PATH / 'results_final.csv')
    df = df[(df['Classifier'] == classifier) & (df['Metric'] == metric)]

    measurements = OrderedDict()

    datasets = df['Dataset'].unique()

    for resampler in RESAMPLERS:
        measurements[resampler] = []

        for dataset in datasets:
            scores = df[(df['Resampler'] == resampler) & (df['Dataset'] == dataset)]['Score']

            assert len(scores) == 10, len(scores)

            measurements[resampler].append(np.mean(scores))

    return measurements


if __name__ == '__main__':
    for classifier in CLASSIFIERS:
        for metric in METRICS:
            if metric == METRICS[0]:
                start = '\\multirow{%d}{*}{%s}' % (len(METRICS), classifier)
            else:
                start = ''

            d = load_final_dict(classifier, metric)
            ranks, _, corrected_p_values = test_friedman_shaffer(d)

            row = [start, metric]

            best_rank = sorted(set(ranks.values()))[0]
            second_best_rank = sorted(set(ranks.values()))[1]

            for resampler in RESAMPLERS:
                rank = ranks[resampler]
                col = '%.2f' % np.round(rank, 2)

                if rank == best_rank:
                    col = '\\textbf{%s}' % col

                if corrected_p_values['RB-CCR-CV'][resampler] <= P_VALUE:
                    if rank < ranks['RB-CCR-CV']:
                        col = '%s \\textsubscript{--}' % col
                    else:
                        col = '%s \\textsubscript{+}' % col

                row.append(col)

            print(' & '.join(row) + ' \\\\')

        if classifier != CLASSIFIERS[-1]:
            print('\\midrule')
