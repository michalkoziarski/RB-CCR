import numpy as np
import pandas as pd

from collections import OrderedDict
from merge import RESULTS_PATH
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr


RESAMPLERS = ['CCR', 'RB-CCR-L', 'RB-CCR-E', 'RB-CCR-H']
CLASSIFIERS = ['CART', 'KNN', 'L-SVM', 'R-SVM', 'P-SVM', 'LR', 'NB', 'R-MLP', 'L-MLP']
METRICS = ['Precision', 'Recall', 'Specificity', 'AUC', 'F-measure', 'G-mean']


def test_friedman_shaffer(dictionary):
    df = pd.DataFrame(dictionary)

    columns = df.columns

    pandas2ri.activate()

    importr('scmamp')

    rFriedmanTest = r['friedmanTest']
    rPostHocTest = r['postHocTest']

    initial_results = rFriedmanTest(df)
    posthoc_results = rPostHocTest(df, test='friedman', correct='shaffer', use_rank=True)

    ranks = np.array(posthoc_results[0])[0]
    p_value = initial_results[2][0]
    corrected_p_values = np.array(posthoc_results[2])

    ranks_dict = {col: rank for col, rank in zip(columns, ranks)}
    corrected_p_values_dict = {}

    for outer_col, corr_p_val_vect in zip(columns, corrected_p_values):
        corrected_p_values_dict[outer_col] = {}

        for inner_col, corr_p_val in zip(columns, corr_p_val_vect):
            corrected_p_values_dict[outer_col][inner_col] = corr_p_val

    return ranks_dict, p_value, corrected_p_values_dict


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
    for metric in METRICS:
        intro = '\\begin{table} \n' \
                '\\small \n' \
                '\\caption{%s.} \n' \
                '\\label{table:regions-%s} \n' \
                '\\centering' % (metric, metric.lower())

        outro = '\\end{table}'

        print(intro)

        for k, classifier_name in enumerate(CLASSIFIERS):
            d = load_final_dict(classifier_name, metric)

            ranks_dict, p_value, corrected_p_values_dict = test_friedman_shaffer(d)

            min_key = min(ranks_dict, key=ranks_dict.get)
            max_key = max(ranks_dict, key=ranks_dict.get)

            rank_strings = []

            method_order = ['RB-CCR-L', 'RB-CCR-E', 'RB-CCR-H', 'CCR']

            for m in method_order:
                rstr = f'{ranks_dict[m]:.2f}'

                if m == min_key:
                    rstr = '\\textbf{' + rstr + '}'

                rank_strings.append(rstr)

            pstrs = []

            for i, m1 in enumerate(method_order):
                prow = []

                for j, m2 in enumerate(method_order):
                    if i == j:
                        prow.append('-')
                    else:
                        prow.append(f'{corrected_p_values_dict[m1][m2]:.3f}')

                pstrs.append(' & '.join(prow))

            body = '\\subfloat[%s]{ \n' \
                   '\\begin{tabularx}{0.32\\textwidth}{lYYYY} \n' \
                   '\\toprule \n' \
                   '& L & E & H & LEH \\\\ \n' \
                   '\\midrule \n' \
                   'Rank & %s \\\\ \n' \
                   '\\midrule \n' \
                   'L & %s \\\\ \n' \
                   'E & %s \\\\ \n' \
                   'H & %s \\\\ \n' \
                   'LEH & %s \\\\ \n' \
                   '\\bottomrule \n' \
                   '\\end{tabularx} \n' \
                   '}' % (classifier_name, ' & '.join(rank_strings), *pstrs)

            print(body)

            if k % 3 == 2 and k % 9 != 8:
                print('')

        print(outro)
        print('')
